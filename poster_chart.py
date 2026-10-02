import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize figure
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(
    50,
    93,
    "INFORMATION ASYMMETRY LOOP",
    fontsize=12,
    fontweight="bold",
    ha="center",
    va="center",
    color="#1A365D",
)

# Box coordinates (width, height)
box_width, box_height = 84, 18
x_pos = 8

boxes = [
    {
        "y": 67,
        "title": "1. SELLER INFORMATION ADVANTAGE",
        "desc": "• Concealed physical & functional defects\n• 20%–30% inflated pricing negotiation anchors",
        "color": "#E2E8F0",
    },
    {
        "y": 38,
        "title": "2. MARKET FRICTION & 'LEMON MARKET'",
        "desc": "• Buyer technical uncertainty & valuation risks\n• Traditional continuous regression fails (R² = 46.1%)",
        "color": "#CBD5E1",
    },
    {
        "y": 9,
        "title": "3. MULTIMODAL AI RESOLUTION",
        "desc": "• NLP Regex extracts hidden condition penalties\n• ResNet18 CNN visually audits chassis damage\n• XGBoost outputs objective Ordinal Tiers",
        "color": "#0F766E",
    },
]

for i, box in enumerate(boxes):
    # Draw rounded box
    rect = patches.FancyBboxPatch(
        (x_pos, box["y"]),
        box_width,
        box_height,
        boxstyle="round,pad=2",
        ec="#334155",
        fc=box["color"],
        lw=1.5,
    )
    ax.add_patch(rect)

    # Box Text
    ax.text(
        x_pos + 4,
        box["y"] + 12,
        box["title"],
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        x_pos + 4,
        box["y"] + 5,
        box["desc"],
        fontsize=8.5,
        color="#334155",
        linespacing=1.3,
    )

    # Draw connecting arrow between boxes (except after the last box)
    if i < len(boxes) - 1:
        arrow = patches.FancyArrowPatch(
            (50, box["y"]),
            (50, box["y"] - 5),
            arrowstyle="->",
            mutation_scale=15,
            color="#0F766E",
            lw=2,
        )
        ax.add_patch(arrow)

plt.tight_layout()
plt.savefig("information_asymmetry_loop.png", dpi=300, bbox_inches="tight")
plt.close()

print("Diagram generated and saved successfully as 'information_asymmetry_loop.png'!")
