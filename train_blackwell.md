# train_blackwell.py — обоснование изменений относительно train.py

## Контекст

GPU: **NVIDIA GeForce RTX 5090** (Blackwell, sm_120).
`train.py` падает с `CUDA error: no kernel image is available for execution on the device`
при попытке загрузить Flash Attention 3 через `kernels`.

---

## Изменение 1: замена Flash Attention 3 на `flex_attention`

**Проблема.**
Библиотека `kernels-community/flash-attn3` не содержит скомпилированных ядер для sm_120.
Проверка в `train.py` обрабатывает только Hopper (sm_90) vs остальные — но «остальные»
подразумевали архитектуры до Blackwell включительно, которые так и не получили поддержку.

**Почему не `F.scaled_dot_product_attention` (SDPA).**
SDPA с параметром `is_causal=True` действительно работает на Blackwell, но для
sliding-window слоёв (паттерн `S`) требует явной булевой маски `(T × T)`:
- O(T²) памяти: 2048² × 1 байт × 6 слоёв ≈ 96 МБ только под маски
- Отключает flash-путь, диспатчится на медленный math backend

**Решение: `torch.nn.attention.flex_attention`.**
- Компилируется через Triton → работает на любой CUDA-архитектуре, включая sm_120
- Принимает `BlockMask` — разреженное представление паттерна внимания
- O(T × window) памяти вместо O(T²) для sliding-window
- Интегрируется с `torch.compile` без graph break

**Реализация.**
Добавлена функция `make_block_mask(window, seq_len, device)`, которая строит `BlockMask`
для каузального или sliding-window паттерна. Маски вычисляются один раз в `init_weights()`
(когда известен device) и хранятся в `self.block_masks`. Сигнатуры `Block.forward` и
`CausalSelfAttention.forward` изменены: `window_size: tuple` → `block_mask: BlockMask`.

```
# было (train.py)
from kernels import get_kernel
fa3 = get_kernel(repo).flash_attn_interface
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)

# стало (train_blackwell.py)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
y = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=...)
```

**Память sliding-window (T=2048, window=1024, 6 слоёв):**

| Реализация | Память под маски |
|---|---|
| SDPA + bool mask | ~96 МБ |
| flex_attention + BlockMask | ~12 МБ |

---

## Изменение 2: `DEVICE_BATCH_SIZE` 128 → 64

**Проблема.**
При первом forward+backward шаге после компиляции:
```
buf229 = empty_strided_cuda((262144, 512), (512, 1), torch.bfloat16)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.
```
`262144 = 128 × 2048` — активационный буфер одного микробатча в backward pass.
GPU имеет 31.36 ГБ, но к моменту первого шага 30.67 ГБ уже занято (модель + оптимизатор
+ маски + промежуточные буферы компилятора).

**Решение.**
Уменьшить `DEVICE_BATCH_SIZE` вдвое: 128 → 64. `TOTAL_BATCH_SIZE` не меняется,
`grad_accum_steps` вырастает с 2 до 4. Эффективный батч остаётся прежним — ~524K токенов
на шаг оптимизатора.

```python
# DEVICE_BATCH_SIZE = 128  # вызывает OOM на RTX 5090 при backward pass
DEVICE_BATCH_SIZE = 64
```
