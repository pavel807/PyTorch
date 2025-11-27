import os
import sys
import time
import traceback
import json
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 0. НАСТРОЙКИ СРЕДЫ (FIX ДЛЯ AMD WINDOWS)
# ==========================================
# Это предотвращает "тихие вылеты" на картах RX 6xxx/7xxx
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['MIOPEN_DEBUG_COMPILE_ONLY'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Для подробного вывода ошибок

try:
    import tiktoken
except ImportError:
    print("❌ Ошибка: Не установлен модуль tiktoken.")
    print("👉 Выполните: pip install tiktoken")
    input("Нажмите Enter для выхода...")
    sys.exit()

# ==========================================
# 1. КОНФИГУРАЦИЯ НЕЙРОСЕТИ
# ==========================================
# Я поставил безопасные настройки для RX 6750 XT
BATCH_SIZE = 4       # Количество диалогов за раз
BLOCK_SIZE = 128     # Длина памяти (контекста)
LEARNING_RATE = 3e-4
EMBED_DIM = 384      # Размерность скрытого слоя
NUM_HEADS = 6        # ВАЖНО: 384 / 6 = 64. (Должно делиться нацело!)
NUM_LAYERS = 6       # Количество слоев (глубина)
DROPOUT = 0.1
EPOCHS = 3           # Сколько раз прогнать весь датасет
MODEL_PATH = "my_full_gpt.pt"
DATA_FILE = "alpaca_data.json"

# ==========================================
# 2. ОПРЕДЕЛЕНИЕ УСТРОЙСТВА
# ==========================================
def get_device():
    if torch.cuda.is_available():
        # На Windows ROCm представляется как CUDA
        print(f"✅ Устройство: GPU ({torch.cuda.get_device_name(0)})")
        return 'cuda'
    elif torch.backends.mps.is_available():
        print("✅ Устройство: Apple Silicon (MPS)")
        return 'mps'
    else:
        print("⚠️ Устройство: CPU (Будет медленно)")
        return 'cpu'

DEVICE = get_device()

# ==========================================
# 3. ДАТАСЕТ
# ==========================================
class AlpacaDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=BLOCK_SIZE):
        if not os.path.exists(json_file):
            print("📥 Скачиваю датасет Alpaca...")
            url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
            urllib.request.urlretrieve(url, json_file)
        
        print("📂 Загрузка JSON...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.tokenizer = tokenizer
        self.samples = []
        
        print(f"⚙️ Обработка {len(data)} диалогов...")
        for item in data: 
            text = f"User: {item['instruction']} {item['input']}\nBot: {item['output']}<|endoftext|>"
            self.samples.append(text)

        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [50256] * (self.max_length - len(tokens))
        
        data = torch.tensor(tokens, dtype=torch.long)
        return data[:-1], data[1:]

# ==========================================
# 4. АРХИТЕКТУРА МОДЕЛИ (GPT)
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
    def forward(self, x): return self.net(x)

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
        # device=idx.device защищает от ошибок MPS/CUDA
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
# 5. ФУНКЦИЯ ОБУЧЕНИЯ
# ==========================================
def train_model():
    print("\n--- 🚀 ЗАПУСК РЕЖИМА ОБУЧЕНИЯ ---")
    
    # Проверка математики
    if EMBED_DIM % NUM_HEADS != 0:
        print(f"❌ ОШИБКА КОНФИГУРАЦИИ: EMBED_DIM ({EMBED_DIM}) не делится на NUM_HEADS ({NUM_HEADS})!")
        return

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = AlpacaDataset(DATA_FILE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BabyGPT(vocab_size=50304).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"🧠 Параметров модели: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    model.train()
    start_time = time.time()

    try:
        for epoch in range(EPOCHS):
            print(f"\n🌀 Эпоха {epoch+1}/{EPOCHS}")
            for i, (xb, yb) in enumerate(dataloader):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                
                logits, loss = model(xb, yb)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"\rStep {i} | Loss: {loss.item():.4f} | Time: {elapsed:.0f}s", end="")
        
        print("\n\n✅ Обучение завершено!")
        print(f"💾 Сохраняю в {MODEL_PATH}...")
        torch.save(model.state_dict(), MODEL_PATH)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n💀 ОШИБКА ПАМЯТИ (OOM)! Уменьшите BATCH_SIZE.")
        else:
            print(f"\n❌ ОШИБКА ВО ВРЕМЯ ОБУЧЕНИЯ: {e}")
            traceback.print_exc()

# ==========================================
# 6. ФУНКЦИЯ ЧАТА
# ==========================================
def chat_with_bot():
    print("\n--- 🤖 ЗАПУСК РЕЖИМА ЧАТА ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Файл {MODEL_PATH} не найден! Сначала обучите модель.")
        return

    tokenizer = tiktoken.get_encoding("gpt2")
    model = BabyGPT(vocab_size=50304).to(DEVICE)
    
    print("📥 Загрузка весов...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
        
    model.eval()
    print("✅ Бот готов! (Напишите 'exit' для выхода)")

    while True:
        user_text = input("\nUser: ")
        if user_text.lower() in ['exit', 'quit']: break
        
        prompt = f"User: {user_text}\nBot:"
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
        
        print("Bot: ", end="", flush=True)
        with torch.no_grad():
            for _ in range(100):
                idx_cond = idx[:, -BLOCK_SIZE:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                token = idx_next.item()
                if token == 50256: break # Конец текста
                
                print(tokenizer.decode([token]), end="", flush=True)
                idx = torch.cat((idx, idx_next), dim=1)
        print()

# ==========================================
# 7. ГЛАВНОЕ МЕНЮ
# ==========================================
if __name__ == '__main__':
    try:
        print("="*40)
        print("   MASTER AI: ROCm/MPS GPT TRAINER")
        print("="*40)
        print("1. 🏋️  Обучить новую модель")
        print("2. 💬  Болтать с уже обученной")
        
        choice = input("\nВаш выбор (1 или 2): ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            chat_with_bot()
        else:
            print("Неверный выбор.")
            
    except Exception as e:
        print("\n❌ ПРОИЗОШЛА НЕПРЕДВИДЕННАЯ ОШИБКА:")
        traceback.print_exc()
        
    print("\nНажмите Enter, чтобы закрыть...")
    input()
