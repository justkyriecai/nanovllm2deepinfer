# 封面生成提示词

已生成，见 `assets/cover.jpg`（2000×667）。原始要求：，横幅比例 3:1（例如 2400×800），不含任何文字。

## 提示词（英文，适用于 Midjourney / Flux / DALL·E / Imagen）

```
An impressionist landscape in the style of Claude Monet, visible loose brushstrokes, soft dappled morning light.
A narrow brook in the foreground winds between stepping stones and water lilies, then widens into a broad luminous river
that flows toward the horizon, where a vast glowing structure of layered arches rises out of the mist like Rouen Cathedral
at dawn — the far structure painted as a lattice of light, hinting at a city of pipes and towers without any literal machinery.
Along the river, small haystack-like mounds mark stages of the journey, each one slightly larger and brighter than the last.
Palette: pale blue, lavender, sage green, warm gold; pastel sky with pink haze. Wide panoramic composition, 3:1 banner,
no text, no letters, no logos, no people.
```

Midjourney 可加：`--ar 3:1 --style raw --stylize 250`

## 负面提示（支持的工具再加）

```
text, watermark, signature, letters, logo, photo-realistic, digital rendering, sharp edges, neon, circuit board, robot
```

## 意象对照

- 小溪 → nano-vllm；大河 → 演进的路线；远处的光之建筑 → DeepInfer
- 踏脚石和逐渐变大的草垛 → 一节一节的增量 PR（草垛取自莫奈的系列画法：同一主题反复画，每次多一点光）
- 睡莲 → 莫奈标志性元素，不解释

## 备选构图

同一条河，分成三联画：清晨（小溪）、正午（河）、黄昏（河口与光之建筑）。更贴近"系列画"，但 3:1 内放三幅会挤，适合 README 之外的宣传图。
