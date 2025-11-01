# WanVideo VACE Encode 4090

优化版的 VACE 编码节点，专门针对 4090 级别显卡优化，提高分块编码性能和显存利用率。

## 主要优化

1. **智能分块大小调整**: 根据输入尺寸自动计算最优分块大小
2. **可调节分块参数**: 支持分块大小乘数和最小分块尺寸设置
3. **优化的内存管理**: 减少小分块造成的性能损失
4. **高效插值算法**: 使用 PyTorch 原生插值函数提高处理速度

## 参数说明

- `tile_size_multiplier`: 分块大小乘数 (0.5-4.0)，越大分块越大，速度越快但占用更多显存
- `min_tile_size`: 最小分块尺寸 (64-1024)，确保分块不会太小影响性能

## 使用建议

对于 4090D 显卡（24GB 显存）:
- 分辨率 832x480: 设置 `tile_size_multiplier=2.0`, `min_tile_size=256`
- 分辨率 1280x720: 设置 `tile_size_multiplier=1.5`, `min_tile_size=192`
- 分辨率 1920x1080: 设置 `tile_size_multiplier=1.0`, `min_tile_size=128`

## 兼容性

完全兼容原 `WanVideoVACEEncode` 节点的所有输入输出，可以直接替换使用。