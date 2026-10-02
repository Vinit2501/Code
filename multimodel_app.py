import streamlit as st, pandas as pd, joblib, torch, torch.nn as nn, time, re
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="UAE Valuation Engine", page_icon="💻", layout="wide")
st.markdown(
    """<style>.stApp {background: radial-gradient(circle at 10% 20%, rgb(15,23,42) 0%, rgb(9,14,23) 100%); color: #f8fafc;} h1 {background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;} header {visibility: hidden;}</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engines():
    return joblib.load("valuation_engine.joblib"), joblib.load("model_columns.joblib")


@st.cache_resource
def load_vision():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load(
            "vision_weights.pth", map_location=torch.device("cpu"), weights_only=True
        )
    )
    return model.eval(), transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


xgb_model, xgb_cols = load_engines()
vis_model, vis_transform = load_vision()

st.title("UAE Secondary Market Valuation Engine")
st.divider()

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("⚙️ 1. Hardware Specifications")
    brand = st.selectbox(
        "Brand",
        ["Apple", "MSI", "Razer", "Lenovo", "Dell", "HP", "Asus", "Acer", "Other"],
    )
    cpu = st.selectbox(
        "Processor",
        [
            "Apple M1/M2/M3",
            "Intel Core i9 / Ryzen 9",
            "Intel Core i7 / Ryzen 7",
            "Intel Core i5 / Ryzen 5",
            "Standard/Other",
        ],
    )
    ram = st.number_input("RAM (GB)", 4, 128, 16, 4)
    storage = st.number_input("Storage (GB)", 128, 8000, 512, 128)
    screen = st.selectbox(
        "Screen Size", ["13-inch", "14-inch", "15.6-inch", "16-inch", "17-inch"]
    )
    gpu = st.selectbox(
        "Dedicated GPU Tier",
        [
            "Tier 4 (RTX 4090 / Mac Max)",
            "Tier 3 (RTX 4070 / Mac Pro)",
            "Tier 2 (RTX 3060)",
            "Tier 1 (GTX 1650)",
            "Tier 0 (Integrated / None)",
        ],
        index=4,
    )

with c2:
    st.subheader("👁️ 2. Unstructured Appraisal")
    if "desc" not in st.session_state:
        st.session_state.desc = ""
    b1, b2, b3 = st.columns(3)
    if b1.button("✨ Mint Condition"):
        st.session_state.desc = (
            "Perfect condition, sealed in box with working keyboard."
        )
    if b2.button("⚠️ Broken Screen"):
        st.session_state.desc = "Urgent sale. Laptop works fine but has a cracked screen and heavy scratches."
    if b3.button("🔋 No Charger"):
        st.session_state.desc = (
            "Selling my old Laptop. Good condition but selling without charger."
        )

    text = st.text_area("Seller Description:", height=100, value=st.session_state.desc)
    img = st.file_uploader("Upload Hardware Image:", type=["jpg", "jpeg", "png"])

if st.button(
    "🚀 Execute Multimodal Valuation", type="primary", use_container_width=True
):
    pb = st.progress(0)
    with st.expander("🔍 System Architecture Execution Logs", expanded=True):
        cv_dmg = 0
        if img:
            st.write("🔄 **Computer Vision:** Analyzing pixel matrix...")
            with torch.no_grad():
                out = vis_model(
                    vis_transform(Image.open(img).convert("RGB")).unsqueeze(0)
                )
                probs, pred = (
                    torch.nn.functional.softmax(out, dim=1)[0],
                    torch.argmax(out).item(),
                )
            if pred == 1:
                cv_dmg = 1
                st.error(
                    f"⚠️ **CNN Alert:** Structural damage detected ({probs[1] * 100:.1f}%)."
                )
            else:
                st.success(
                    f"✅ **CNN Cleared:** Chassis passes inspection ({probs[0] * 100:.1f}%)."
                )
        else:
            st.info("ℹ️ **Computer Vision:** Skipped (No image).")
        pb.progress(33)

        st.write("🔄 **NLP Pipeline:** Parsing unstructured text...")
        t = str(text).lower()
        feats = {
            "Damaged_Display": 1
            if re.search("cracked|broken screen|dead pixel|lines", t)
            else 0,
            "Body_Damage": 1
            if cv_dmg
            or re.search("dented|heavy scratches|missing keys|broken hinge", t)
            else 0,
            "Missing_Charger": 1
            if re.search("no charger|without charger|laptop only", t)
            else 0,
            "Perfect_Condition": 1
            if re.search("brand new|sealed|unopened|mint", t)
            else 0,
            "Arabic_Keyboard": 1 if re.search("arabic|gcc", t) else 0,
        }
        hits = [
            k.replace("_", " ")
            for k, v in feats.items()
            if v == 1 and k != "Body_Damage"
        ]
        if cv_dmg:
            hits.append("Body Damage (Vision)")
        if hits:
            st.success(f"📝 **Extracted Vectors:** {', '.join(hits)}")
        else:
            st.info("📝 **Extracted Vectors:** No explicit conditions found.")
        pb.progress(66)

        st.write("🔄 **Data Engineering:** Triggering XGBoost Cascade...")

        # UI to ML inputs
        seg_map = {
            "apple": "Premium_Ecosystem",
            "msi": "Premium_Gaming",
            "razer": "Premium_Gaming",
            "lenovo": "Mainstream_Corporate",
            "dell": "Mainstream_Corporate",
            "hp": "Mainstream_Corporate",
            "asus": "Mainstream_Consumer",
            "acer": "Mainstream_Consumer",
        }
        m_seg = seg_map.get(brand.lower(), "Standard_Other")

        c_tier = (
            "apple_silicon"
            if "M1" in cpu
            else (
                "i9"
                if "i9" in cpu
                else ("i7" if "i7" in cpu else ("i5" if "i5" in cpu else "standard"))
            )
        )
        s_size = (
            "large" if screen in ["15.6-inch", "16-inch", "17-inch"] else "standard"
        )
        g_tier = (
            4
            if "Tier 4" in gpu
            else (
                3
                if "Tier 3" in gpu
                else (2 if "Tier 2" in gpu else (1 if "Tier 1" in gpu else 0))
            )
        )

        df_in = pd.DataFrame(
            [
                {
                    "RAM_GB": ram,
                    "Storage_GB": storage,
                    "GPU_Tier": g_tier,
                    **feats,
                    "Market_Segment": m_seg,
                    "CPU_Tier": c_tier,
                    "Screen_Size": s_size,
                }
            ]
        )

        X = pd.concat(
            [df_in.iloc[:, :8], pd.get_dummies(df_in.iloc[:, 8:])], axis=1
        ).reindex(columns=xgb_cols, fill_value=0)

        pred_tier = xgb_model.predict(X)[0]
        pb.progress(100)

        tiers = {
            0: ("Budget Tier", "Under 1,500 AED", "⬇️"),
            1: ("Mid-Range Tier", "1,500 - 4,000 AED", "⚖️"),
            2: ("Premium Tier", "Over 4,000 AED", "💎"),
        }
        t_name, p_range, emoji = tiers[pred_tier]

        tier_probs = xgb_model.predict_proba(X)

        budget_median = 1000
        mid_median = 2750
        premium_median = 5500

        exact_price = (
            (tier_probs[0][0] * budget_median)
            + (tier_probs[0][1] * mid_median)
            + (tier_probs[0][2] * premium_median)
        )

        # FOOLPROOF DAMAGE PENALTY
        damage_found = False
        for col in X.columns:
            if "Damage" in col or "Missing" in col:
                if X[col].iloc[0] == 1:
                    damage_found = True

        if damage_found:
            exact_price = (
                exact_price * 0.55
            )  # 45% massive price drop for the broken laptop

        specific_price_str = f"{int(exact_price):,} AED"
        narrow_range_str = (
            f"{int(exact_price * 0.95):,} - {int(exact_price * 1.05):,} AED"
        )

        # UI
        st.subheader("Algorithmic Appraisal Result 🔗")
        col1, col2, col3 = st.columns(3)

        col1.metric("Market Classification", f"{emoji} {t_name}")
        col2.metric("Specific Estimated Value", specific_price_str)
        col3.metric("Fair Negotiation Range", narrow_range_str)

        # Confidence Score
        max_prob = max(tier_probs[0])
        st.caption(f"**AI Confidence Score:** {max_prob * 100:.1f}%")
        st.progress(float(max_prob))
