# Circuit Explorer Style Guide

> **For future agents working on this dashboard.**
> This document is the single source of truth for all visual design decisions.

---

## Design Philosophy

**Functional Minimalism** — Clean, spacious, research-grade interfaces.
No gratuitous gradients, no glows, no "AI slop." Every element earns its presence.

The aesthetic is AIcrowd's: a white-canvas platform for serious research work,
punctuated by sharp Coral Red accents on interactive elements.

---

## Color Palette

### Core Colors

| Token | Hex | Usage |
|---|---|---|
| `--coral` | `#F0524D` | Primary action: buttons, active tabs, links, interactive highlights |
| `--coral-hover` | `#E04440` | Hover state for coral elements |
| `--coral-light` | `#FEF2F1` | Subtle coral tint for backgrounds (active states, callouts) |

### Neutrals

| Token | Hex | Usage |
|---|---|---|
| `--white` | `#FFFFFF` | Page background, card backgrounds |
| `--gray-50` | `#F8F9FA` | Section backgrounds, alternate rows, sidebar |
| `--gray-100` | `#F1F3F5` | Input backgrounds, subtle dividers |
| `--gray-200` | `#E0E0E0` | Borders, separators |
| `--gray-400` | `#9CA3AF` | Muted text, placeholders, disabled elements |
| `--gray-600` | `#5D5F60` | Body text, secondary labels |
| `--gray-900` | `#1F1F1F` | Headings, primary text |

### Data Visualization

| Token | Hex | Usage |
|---|---|---|
| `--data-blue` | `#3B82F6` | Sampling estimator, primary data series |
| `--data-amber` | `#F59E0B` | Mean propagation estimator, secondary data series |
| `--data-emerald` | `#10B981` | Success states, positive indicators |
| `--signal-neg` | `#3B82F6` | Wire mean = −1 (blue) |
| `--signal-zero` | `#9CA3AF` | Wire mean ≈ 0 (gray) |
| `--signal-pos` | `#F0524D` | Wire mean = +1 (coral) |

---

## Typography

### Font Stack

```css
--font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
--font-mono: 'IBM Plex Mono', 'SF Mono', Consolas, monospace;
```

**DM Sans** is the primary typeface — geometric, modern, and highly legible at all sizes.
**IBM Plex Mono** for data values, code, and technical labels.

### Scale

| Role | Size | Weight | Line-height | Tracking |
|---|---|---|---|---|
| Page title | 20px | 700 | 1.2 | −0.02em |
| Section heading | 11px | 600 | 1.4 | 0.08em (uppercase) |
| Body text | 13px | 400 | 1.5 | 0 |
| Data value | 12px | 500 | 1.0 | 0 (mono) |
| Caption/help | 11px | 400 | 1.4 | 0 |
| Tiny label | 10px | 500 | 1.0 | 0.04em |

### Rules

- **Section headings** are always UPPERCASE, letter-spaced, 11px, `--gray-400` color.
- **Never** use bold for body text. Use color or spacing to create hierarchy.
- Data values (slider readouts, axis labels) always use `--font-mono`.

---

## Spacing

Base unit: **4px**. All spacing is a multiple of 4.

| Context | Value |
|---|---|
| Component internal padding | 16px |
| Card/panel padding | 20px |
| Gap between panels | 16px |
| Section margin | 24px |
| Sidebar width | 260px |

---

## Components

### Buttons

```
Primary:    bg: --coral, text: white, radius: 20px, padding: 8px 20px
            hover: --coral-hover, translateY(-1px), subtle shadow
Secondary:  bg: transparent, border: 1px --gray-200, text: --gray-600, radius: 20px
            hover: bg --gray-50, border --gray-400
```

**Pill-shaped** (border-radius ≥ 20px). No sharp corners on interactive elements.

### Input Fields

```
bg: --white, border: 1px --gray-200, radius: 8px, padding: 6px 10px
focus: border --coral, box-shadow: 0 0 0 3px --coral-light
```

### Sliders

```
Track:  bg: --gray-200, height: 3px, radius: 2px
Thumb:  bg: --coral, 14px circle, border: 2px --white
        hover: scale(1.15)
```

### Cards / Panels

```
bg: --white, border: 1px --gray-200, radius: 10px
No box-shadow by default. Shadow only on hover/focus if needed.
```

### Tabs / Step Controls

Active state uses a `2px` bottom border in `--coral`, text in `--coral`.
Inactive tabs: `--gray-400` text, no border.

---

## Data Visualization Rules

### Circuit Graph (SVG)

- **Wire connections**: Use `stroke-width: 1.2`, muted opacity (0.4–0.6).
  First input = solid line. Second input = dashed (`strokeDasharray: 3,3`).
- **Gate nodes**: 10px radius circles, filled with signal color.
  Active layer gets `stroke: --coral, strokeWidth: 2`.
- **Input labels**: `--gray-400`, 10px, mono font.
- **Layer labels**: Centered below each column, `--gray-400`, 10px.

### Heatmap (Canvas)

- Color range: `--signal-neg` (blue) → `--signal-zero` (gray) → `--signal-pos` (coral).
- 1px gap between cells. Rounded corners on the canvas container.
- Always include axis labels ("Layer ↓", "Wire →").

### Comparison Charts (Canvas)

- Bar charts with `--data-blue` and `--data-amber`.
- Grid lines: `--gray-100`, 1px.
- Axis text: `--gray-400`, 10px mono.
- Legend: small colored squares + 11px labels, top-right of chart.

---

## Layout

```
┌──────────────────────────────────────────────┐
│ Header: logo + title (left) ─ clean, minimal │
├──────────┬───────────────────────────────────┤
│ Sidebar  │ Main Content                      │
│ bg:gray50│ bg:white                          │
│ 260px    │ flex: 1                           │
│          │ ┌───────────────────────────────┐  │
│ Controls │ │ Circuit Graph (full width)    │  │
│          │ └───────────────────────────────┘  │
│ Stepper  │ ┌──────────┐ ┌────────────────┐  │
│          │ │ Heatmap   │ │ Comparison     │  │
│          │ └──────────┘ └────────────────┘  │
├──────────┴───────────────────────────────────┤
│ Footer: minimal, --gray-400 text, 11px       │
└──────────────────────────────────────────────┘
```

- Sidebar has a subtle `border-right: 1px --gray-200`.
- Main content scrolls independently.
- Bottom panels are a 2-column grid with `gap: 16px`.
- On screens < 900px, sidebar stacks above main content.

---

## Do's and Don'ts

### ✅ Do

- Use whitespace generously — let content breathe
- Keep interactive elements pill-shaped
- Use coral ONLY for actions and active states
- Use data colors consistently (blue = sampling, amber = mean-prop)
- Use mono font for all numerical values
- Keep section headings uppercase and letter-spaced

### ❌ Don't

- Use dark backgrounds (this is a light-themed tool)
- Add gradients, glows, or glassmorphism
- Use more than 2 data colors in a single chart
- Make text smaller than 10px
- Add drop shadows to cards
- Use any color outside the defined palette
