import pandas as pd
import re


def clean_data():
    print("Loading deep-scraped dataset...")
    df = pd.read_csv("dubizzle_laptops_deep.csv")

    # Price Outliers
    df = df.dropna(subset=["Title", "Price (AED)"])
    df["Price (AED)"] = pd.to_numeric(
        df["Price (AED)"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    df = df[(df["Price (AED)"] >= 300) & (df["Price (AED)"] <= 30000)].copy()

    text_series = (
        df["Title"].fillna("") + " " + df["Description"].fillna("")
    ).str.lower()

    # V2 Architecture Mappings
    df["Market_Segment"] = text_series.apply(
        lambda t: (
            "Premium_Ecosystem"
            if any(b in t for b in ["macbook", "apple"])
            else (
                "Premium_Gaming"
                if any(b in t for b in ["alienware", "razer", "msi", "rog", "legion"])
                else (
                    "Mainstream_Corporate"
                    if any(
                        b in t
                        for b in [
                            "thinkpad",
                            "latitude",
                            "elitebook",
                            "lenovo",
                            "dell",
                            "hp",
                        ]
                    )
                    else (
                        "Mainstream_Consumer"
                        if any(b in t for b in ["asus", "acer", "vivobook", "ideapad"])
                        else (
                            "Budget_Entry"
                            if any(b in t for b in ["chuwi", "ilife", "celeron"])
                            else "Standard_Other"
                        )
                    )
                )
            )
        )
    )

    df["GPU_Tier"] = text_series.apply(
        lambda t: (
            4
            if re.search(r"rtx\s*(4080|4090)|rx\s*7900|m[23]\s*max", t)
            else (
                3
                if re.search(r"rtx\s*(3080|3090|4070)|rx\s*6800|m[123]\s*pro", t)
                else (
                    2
                    if re.search(r"rtx\s*(2070|3060|3070|4050|4060)|rx\s*6600", t)
                    else (
                        1
                        if re.search(
                            r"gtx\s*(1650|1660|1050|1060)|rtx\s*2060|rx\s*580", t
                        )
                        else 0
                    )
                )
            )
        )
    )

    df["CPU_Tier"] = text_series.apply(
        lambda t: (
            "apple_silicon"
            if re.search(r"m[123]", t)
            else (
                "i9"
                if re.search(r"i9|ryzen\s*9", t)
                else (
                    "i7"
                    if re.search(r"i7|ryzen\s*7", t)
                    else ("i5" if re.search(r"i5|ryzen\s*5", t) else "standard")
                )
            )
        )
    )

    df["Screen_Size"] = text_series.apply(
        lambda t: (
            "large"
            if any(s in t for s in ['15"', "15.", '16"', "16.", '17"'])
            else "standard"
        )
    )

    df["RAM_GB"] = df["Title"].apply(
        lambda t: (
            int(m.group(1))
            if (
                m := re.search(
                    r"\b(\d{1,3})\s*(?:GB|gb)\b(?!\s*(?:SSD|HDD))",
                    str(t),
                    re.IGNORECASE,
                )
            )
            else 16
        )
    )
    df["Storage_GB"] = df["Title"].apply(
        lambda t: (
            int(m.group(1)) * 1024
            if (m := re.search(r"\b(1|2)\s*(?:TB|tb)\b", str(t), re.IGNORECASE))
            else (
                int(m2.group(1))
                if (
                    m2 := re.search(
                        r"\b(\d{1,4})\s*(?:GB|TB|gb|tb)?\s*(?:SSD|HDD|NVMe|Storage)\b",
                        str(t),
                        re.IGNORECASE,
                    )
                )
                else 512
            )
        )
    )

    df["Damaged_Display"] = text_series.str.contains(
        "cracked|broken screen|dead pixel|lines on screen", regex=True
    ).astype(int)
    df["Body_Damage"] = text_series.str.contains(
        "dented|heavy scratches|missing keys|broken hinge", regex=True
    ).astype(int)
    df["Missing_Charger"] = text_series.str.contains(
        "no charger|without charger|laptop only", regex=True
    ).astype(int)
    df["Perfect_Condition"] = text_series.str.contains(
        "brand new|sealed|unopened|mint", regex=True
    ).astype(int)
    df["Arabic_Keyboard"] = text_series.str.contains("arabic|gcc", regex=True).astype(
        int
    )

    df["Market_Tier"] = df["Price (AED)"].apply(
        lambda x: 0 if x < 1500 else (1 if x <= 4000 else 2)
    )

    cols = [
        "Price (AED)",
        "Market_Tier",
        "Market_Segment",
        "CPU_Tier",
        "GPU_Tier",
        "Screen_Size",
        "RAM_GB",
        "Storage_GB",
        "Damaged_Display",
        "Body_Damage",
        "Missing_Charger",
        "Perfect_Condition",
        "Arabic_Keyboard",
    ]
    df[cols].to_csv("dubizzle_laptops_cleaned.csv", index=False)
    print("SUCCESS! Saved updated dubizzle_laptops_cleaned.csv")


if __name__ == "__main__":
    clean_data()
