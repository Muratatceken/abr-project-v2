# 🔧 Attention Dimension Compatibility Fix

## 🐛 **Problem**

```
AssertionError: d_model (160) must be divisible by n_heads (12)
```

## 🔍 **Root Cause**

The **Multi-Head Attention** mechanism requires that:
```
d_model % n_heads == 0
```

This is because each attention head gets `d_model / n_heads` dimensions.

## ✅ **Solution Applied**

### **Fixed Configurations**

| Configuration | d_model | n_heads | Valid? | Calculation |
|---------------|---------|---------|--------|-------------|
| **Original** | 160 | 12 | ❌ | 160 ÷ 12 = 13.33... |
| **Fixed v1** | 160 | 8 | ✅ | 160 ÷ 8 = 20 |
| **Fixed v2** | 192 | 12 | ✅ | 192 ÷ 12 = 16 |

### **Changes Made**

#### **configs/config.yaml & configs/config_optimized.yaml**
```yaml
decoder:
  n_heads: 8  # Changed from 12 to 8
```

#### **configs/config_optimized_v2.yaml** (Alternative)
```yaml
model:
  architecture:
    base_channels: 96  # Changed from 80 to make d_model = 192
decoder:
  n_heads: 12  # Can keep 12 heads with larger d_model
```

---

## 📊 **Dimension Calculation**

### **How d_model is Calculated**
```python
d_model = base_channels * 2  # In decoder blocks
```

### **Valid Combinations**

| base_channels | d_model | Valid n_heads |
|---------------|---------|---------------|
| 64 | 128 | 1, 2, 4, 8, 16, 32, 64 |
| 80 | 160 | 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80 |
| 96 | 192 | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96 |
| 128 | 256 | 1, 2, 4, 8, 16, 32, 64, 128 |

### **Recommended Combinations**

| Model Size | base_channels | d_model | n_heads | Head Dimension |
|------------|---------------|---------|---------|----------------|
| **Small** | 64 | 128 | 8 | 16 |
| **Medium** | 80 | 160 | 8 | 20 |
| **Large** | 96 | 192 | 12 | 16 |
| **XLarge** | 128 | 256 | 16 | 16 |

---

## 🚀 **Available Configurations**

### **Option 1: Fixed Original (Recommended)**
```bash
python train.py --config configs/config.yaml --experiment fixed_standard
```
- **d_model**: 160, **n_heads**: 8
- **Conservative** but reliable
- **Good performance** with stability

### **Option 2: Enhanced Optimized**
```bash
python train.py --config configs/config_optimized.yaml --experiment fixed_optimized
```
- **d_model**: 160, **n_heads**: 8  
- **All optimizations** applied
- **Best balance** of performance and stability

### **Option 3: Maximum Performance**
```bash
python train.py --config configs/config_optimized_v2.yaml --experiment max_performance
```
- **d_model**: 192, **n_heads**: 12
- **Largest model** with most attention heads
- **Maximum capacity** for complex tasks

---

## 🔍 **Why This Happens**

### **Multi-Head Attention Mechanism**
```python
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0  # THIS CHECK
        self.head_dim = d_model // n_heads  # Must be integer
```

### **Head Dimension Calculation**
Each attention head operates on `head_dim = d_model / n_heads` dimensions:
- **Head 1**: dimensions [0:head_dim]
- **Head 2**: dimensions [head_dim:2*head_dim]  
- **Head N**: dimensions [(N-1)*head_dim:N*head_dim]

If `d_model` is not divisible by `n_heads`, the dimensions don't split evenly.

---

## 🛠️ **Quick Fix Reference**

### **If You Get This Error**
1. **Check your config** for `n_heads` value
2. **Calculate d_model** = base_channels × 2
3. **Choose compatible n_heads** from the table above
4. **Update config** and restart training

### **Common Valid n_heads Values**
- **For d_model=128**: 1, 2, 4, 8, 16
- **For d_model=160**: 1, 2, 4, 5, 8, 10, 16, 20
- **For d_model=192**: 1, 2, 3, 4, 6, 8, 12, 16, 24
- **For d_model=256**: 1, 2, 4, 8, 16, 32

### **Performance Recommendations**
- **8 heads**: Good balance for most tasks
- **12 heads**: Better for complex attention patterns  
- **16 heads**: Maximum attention capacity

---

## ✅ **Verification**

Test the fix:
```bash
python train.py --config configs/config.yaml --fast_dev_run --epochs 1 --batch_size 4
```

You should see:
```
✓ Model created successfully
✓ No dimension errors
✓ Training starts normally
```

---

**🎉 The attention dimension compatibility issue is now fixed in all configurations!**