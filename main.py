import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import json
import tiktoken
import os
import urllib.request
import time

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
# Увеличим параметры, так как теперь мы серьезны
BATCH_SIZE = 8       # Если вылетает ошибка памяти (OOM), уменьшите до 4 или 2
BLOCK_SIZE = 128     # Длина контекста
LEARNING_RATE = 3e-4
EMBED_DIM = 384      # Чуть больше "мозгов" (было 256)
NUM_HEADS = 6        # 384 / 6 = 64 размер головы
NUM_LAYERS = 6       # Больше слоев для глубины
DROPOUT = 0.1
EPOCHS = 3           # Пройдем по всему датасету 3 раза

# Название файла для сохранения модели
MODEL_PATH = "my_alpaca_gpt.pt"

# --- ЛОГИКА ВЫБОРА УСТРОЙСТВА (CUDA / ROCm / MPS / CPU) ---
def get_device():
    # 1. Проверяем CUDA (NVIDIA) или ROCm (AMD)
    # PyTorch для ROCm использует интерфейс 'cuda', поэтому is_available() вернет True
    if torch.cuda.is_available():
        # Проверим, это AMD или Nvidia
        if torch.version.hip:
            print(f"✅ Устройство: AMD GPU (ROCm) | {torch.cuda.get_device_name(0)}")
        else:
            print(f"✅ Устройство: NVIDIA GPU (CUDA) | {torch.cuda.get_device_name(0)}")
        return 'cuda'
    
    # 2. Проверяем Apple Metal (Mac M1/M2/M3)
    elif torch.backends.mps.is_available():
        print("✅ Устройство: Apple Silicon (MPS/Metal)")
        return 'mps'
    
    # 3. Fallback на процессор
    else:
        print("⚠️ Устройство: CPU (Внимание: обучение будет очень медленным)")
        return 'cpu'

DEVICE = get_device()
# ---------------------------------------------------

# ==========================================
# 2. ПОДГОТОВКА ДАТАСЕТА
# ==========================================
class AlpacaDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=BLOCK_SIZE):
        if not os.path.exists(json_file):
            print("⏳ Скачиваю alpaca_data.json...")
            url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
            urllib.request.urlretrieve(url, json_file)
        
        print("⏳ Загрузка и обработка JSON...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.tokenizer = tokenizer
        self.samples = []
        
        # Используем ВЕСЬ датасет (52k примеров)
        print(f"Всего диалогов в файле: {len(data)}. Подготовка токенов...")
        
        for item in data:
            # Формируем строку обучения
            # Добавляем явный промпт для модели
            text = f"User: {item['instruction']} {item['input']}\nBot: {item['output']}<|endoftext|>"
            self.samples.append(text)

        print(f"✅ Датасет готов. Примеров: {len(self.samples)}")
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        # Токенизация
        tokens = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        
        # Обрезка или паддинг
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # 50256 - это токен <|endoftext|> в GPT-2
            tokens = tokens + [50256] * (self.max_length - len(tokens))
            
        data = torch.tensor(tokens, dtype=torch.long)
        x = data[:-1]
        y = data[1:]
        return x, y

# ==========================================
# 3. АРХИТЕКТУРА МОДЕЛИ (GPT)
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BabyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[Block(EMBED_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # device=idx.device гарантирует, что эмбеддинги создаются там же, где данные (MPS/CUDA)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# ==========================================
# 4. ЗАПУСК
# ==========================================
if __name__ == '__main__':
    torch.manual_seed(1337)
    
    # 1. Токенизатор
    print("Инициализация токенизатора...")
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except:
        print("Ошибка: не установлен tiktoken. Выполните pip install tiktoken")
        exit()
    
    # 2. Датасет и Dataloader
    dataset = AlpacaDataset('alpaca_data.json', tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Модель
    print(f"Создание модели (Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}, Emb: {EMBED_DIM})...")
    model = BabyGPT(vocab_size=50304) # 50304 - стандарт GPT-2 (красиво делится)
    
    # Перенос модели на устройство
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 Модель готова. Количество параметров: {num_params/1e6:.2f} M")
    print(f"🚀 Начинаем обучение на {DEVICE} | Эпох: {EPOCHS}")

    # --- ЦИКЛ ОБУЧЕНИЯ ---
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        for i, (xb, yb) in enumerate(dataloader):
            # Перенос батча на GPU/MPS
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # Forward
            logits, loss = model(xb, yb)
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Логирование
            if i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {i} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
                # Можно сохранять чекпоинты каждые N шагов, если нужно

    print("🏁 Обучение завершено!")
    
    # --- СОХРАНЕНИЕ ---
    print(f"💾 Сохраняю веса модели в {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)

    # ==========================================
    # 5. ТЕСТ ГЕНЕРАЦИИ (INFERENCE)
    # ==========================================
    print("\n--- 🤖 ТЕСТ ЧАТ-БОТА ---")
    model.eval()
    
    # Функция для чистой генерации
    def generate_response(prompt, max_tokens=100):
        full_prompt = f"User: {prompt}\nBot:"
        input_ids = tokenizer.encode(full_prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        
        # Список для сгенерированных токенов
        generated = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Обрезаем контекст, если он стал слишком длинным
                idx_cond = x[:, -BLOCK_SIZE:]
                
                # Получаем предсказание
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                
                # Sampling (выбор с вероятностью)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token_id = idx_next.item()
                
                # Если модель решила закончить фразу (токен <|endoftext|>)
                if token_id == 50256:
                    break
                
                generated.append(token_id)
                # Добавляем к входу для следующего шага
                x = torch.cat((x, idx_next), dim=1)
        
        return tokenizer.decode(generated)

    # Примеры вопросов
    questions = [
        "Hello, how are you?",
        "What is Python?",
        "Tell me a story about a cat."
    ]

    for q in questions:
        print(f"\nUser: {q}")
        ans = generate_response(q)
        print(f"Bot: {ans}")
