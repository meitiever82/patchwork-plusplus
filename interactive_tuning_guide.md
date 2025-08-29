# 🎛️ Patchwork++ 交互式参数调试界面使用指南

## 功能概述

这个交互式调试界面允许你实时调整Patchwork++和网格地图生成的各种参数，立即看到效果变化，快速找到最佳参数组合。

## 运行方式

```bash
# 编译（如果还没有编译）
make cppinstall_with_demo

# 运行交互式调试界面
./cpp/build/examples/demo_interactive_tuning <pcd_file> [param_file]

# 示例：
./cpp/build/examples/demo_interactive_tuning ./data/casbot.101.250730.downsampled.pcd
./cpp/build/examples/demo_interactive_tuning ./data/casbot.101.250730.downsampled.pcd default_params.txt
./cpp/build/examples/demo_interactive_tuning ./data/casbot.101.250730.downsampled.pcd my_custom_params.txt
```

### 📁 参数文件格式
参数文件支持 `key=value` 格式，支持注释：
```
# Patchwork++ Parameters
sensor_height=1.2
th_seeds=0.5
th_dist=0.3
uprightness_thr=0.707
height_threshold=3.5
resolution=0.1
opening_size=2
closing_size=4
dilation_size=1
output_path=interactive_output
```

## 界面布局

启动后会看到三个窗口：

### 1. **Patchwork++ Parameters** (参数控制面板)
包含所有可调参数的滑条控件

### 2. **Grid Map (Real-time)** (实时彩色地图)
- 彩色显示当前参数下的网格地图
- 显示统计信息：地面点数、障碍物点数、处理时间

### 3. **Grid Map (Grayscale)** (灰度地图)
- 灰度显示，便于观察细节
- 黑色=地面，白色=障碍物，灰色=未知

### 🖥️ **界面信息显示**
彩色地图窗口显示：
- **顶部**: `Ground: 24815 | Obstacles: 132598 | Time: 25ms` (统计信息)
- **中间**: `Params: default_params.txt | Output: interactive_output` (文件路径)
- **底部**: 两个**可编辑文本输入框**
  - **Parameter File**: 参数文件路径 (可点击编辑)
  - **Output Path**: 输出路径 (可点击编辑)

### 📝 **文本输入框操作**
- **点击输入框**: 激活编辑模式（蓝色边框）
- **输入文字**: 直接键盘输入
- **回车键**: 确认输入并应用更改
- **ESC键**: 取消输入，恢复原值
- **退格键**: 删除字符

## 🎚️ 参数说明

### **Patchwork++ 核心参数**

| 滑条名称 | 范围 | 默认值 | 说明 |
|----------|------|--------|------|
| **Sensor Height (*100)** | 0-300 | 120 | 传感器高度(米) × 100 |
| **TH Seeds (*100)** | 0-200 | 50 | 地面种子阈值 × 100 |
| **TH Dist (*100)** | 0-100 | 30 | 地面距离阈值 × 100 |
| **Uprightness (*100)** | 0-100 | 71 | 地面法向量阈值 × 100 |

### **网格地图参数**

| 滑条名称 | 范围 | 默认值 | 说明 |
|----------|------|--------|------|
| **Height Threshold (*10)** | 0-100 | 35 | 高度过滤阈值(米) × 10 |
| **Resolution (*100)** | 1-50 | 10 | 网格分辨率(米/像素) × 100 |

### **形态学参数**

| 滑条名称 | 范围 | 默认值 | 说明 |
|----------|------|--------|------|
| **Opening Size** | 0-10 | 2 | 开运算核大小 |
| **Closing Size** | 0-10 | 4 | 闭运算核大小 |
| **Dilation Size** | 0-5 | 1 | 膨胀核大小 |

## ⌨️ 快捷键

| 按键 | 功能 |
|------|------|
| **s** | 保存当前参数到参数文件 |
| **r** | 从参数文件重新加载参数 |
| **w** | 保存当前网格地图到输出目录 |
| **p** | 打印当前所有参数到控制台 |
| **ESC** | 退出程序 |

## 🔧 参数调优策略

### **1. 地面分割优化**

**问题**: 地面检测不准确
- ⬇️ 降低 **TH Seeds** 和 **TH Dist** → 更严格的地面检测
- ⬆️ 提高 **TH Seeds** 和 **TH Dist** → 更宽松的地面检测

**问题**: 倾斜地面检测不到
- ⬇️ 降低 **Uprightness** (如到50) → 接受更倾斜的表面

### **2. 障碍物检测优化**

**问题**: 墙壁顶部缺失
- ⬆️ 提高 **Height Threshold** → 保留更高的点

**问题**: 有太多噪声点
- ⬆️ 增大 **Opening Size** → 移除更多小噪声

### **3. 地面连续性优化**

**问题**: 地面不连续
- ⬆️ 增大 **Closing Size** → 连接更多地面间隙

**问题**: 地面过度扩张
- ⬇️ 减小 **Closing Size** 和 **Dilation Size**

### **4. 实时性能平衡**

**问题**: 处理速度慢
- ⬆️ 增大 **Resolution** → 降低网格分辨率
- ⬇️ 减小形态学核大小

## 📊 文件管理功能

### **参数文件操作**
- **s键**: 保存当前参数到指定的参数文件
- **r键**: 从参数文件重新加载参数（实时更新滑条）
- **p键**: 在控制台打印当前所有参数

### **网格地图保存**
- **w键**: 保存当前网格地图到输出目录
- 自动生成时间戳文件名：`gridmap_<timestamp>.pgm` 和 `gridmap_<timestamp>_vis.png`
- PGM格式兼容ROS，PNG格式用于可视化

### **支持的参数**
保存的参数文件包含：
```
# Patchwork++ Core Parameters
sensor_height=1.2
th_seeds=0.5
th_dist=0.3
th_seeds_v=0.4
th_dist_v=0.3
uprightness_thr=0.707
max_range=80.0
min_range=0.3

# Grid Map Parameters  
height_threshold=3.5
resolution=0.1

# Morphological Parameters
opening_size=2
closing_size=4
dilation_size=1

# Output Parameters
output_path=interactive_output
```

## 🎯 调优工作流程

### **第一步：基础地面分割**
1. 先调整 **Sensor Height** 匹配你的实际传感器
2. 调整 **TH Seeds** 和 **TH Dist** 获得准确的地面检测
3. 观察绿色统计信息中的地面点数变化

### **第二步：障碍物优化**  
1. 调整 **Height Threshold** 确保建筑物完整
2. 观察白色区域是否包含所有重要障碍物

### **第三步：形态学优化**
1. 调整 **Opening Size** 移除噪声
2. 调整 **Closing Size** 改善连续性
3. 微调 **Dilation Size** 设置安全边界

### **第四步：保存最优参数**
1. 按 **s** 保存当前参数
2. 将参数应用到实际代码中

## 💡 使用技巧

1. **实时反馈**: 每次滑条变化后立即看到效果，处理时间显示在彩色地图上
2. **多视角观察**: 同时观察彩色地图和灰度地图
3. **性能监控**: 关注处理时间，平衡精度和速度
4. **参数范围**: 滑条使用整数，有些参数需要除以10或100换算
5. **快速重置**: 按 **r** 可以随时回到默认参数

## 🚀 高级用法

### 批量测试不同数据
```bash
# 测试不同的点云文件
./cpp/build/examples/demo_interactive_tuning ./data/scene1.pcd
./cpp/build/examples/demo_interactive_tuning ./data/scene2.pcd
```

### 参数文件管理
```bash
# 为不同场景保存不同参数
cp tuned_parameters.txt indoor_params.txt
cp tuned_parameters.txt outdoor_params.txt
```

这个交互式界面让参数调优变得直观高效，你可以快速找到适合特定场景的最佳参数组合！