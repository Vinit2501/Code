import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import random


def scrape_deep_descriptions():
    """
    Methodological Pivot:
    Standard requests/BeautifulSoup libraries fail on modern classified platforms (Dubizzle)
    due to Cloudflare bot-mitigation and Imperva WAFs.
    This script utilizes `undetected_chromedriver` to patch the ChromeDriver binary,
    masking the `navigator.webdriver` flag to successfully bypass fingerprinting.
    """
    print("Loading target dataset for deep extraction...")
    df = pd.read_csv("dubizzle_laptops_deep.csv")

    if "Description" not in df.columns:
        df["Description"] = ""

    print("Initializing stealth browser (Cloudflare Bypass Enabled)...")
    options = uc.ChromeOptions()
    options.add_argument("--disable-popup-blocking")

    driver = uc.Chrome(options=options)

    print(f"Deep scraping {len(df)} listings. Initiating stochastic delay protocol...")

    for index, row in df.iterrows():
        if pd.notna(row.get("Description")) and row.get("Description") != "":
            continue

        url = row["Listing_URL"]

        try:
            driver.get(url)

            time.sleep(random.uniform(3.5, 5.5))

            description_elements = driver.find_elements(
                By.XPATH,
                "//div[contains(@data-ui-id, 'description') or contains(@class, 'description')]",
            )

            if description_elements:
                description_text = " ".join(
                    [elem.text for elem in description_elements]
                )
                df.at[index, "Description"] = description_text
                print(
                    f"Row {index}: Successfully extracted {len(description_text)} characters."
                )
            else:
                p_tags = driver.find_elements(By.TAG_NAME, "p")
                description_text = " ".join(
                    [p.text for p in p_tags if len(p.text) > 20]
                )
                df.at[index, "Description"] = description_text
                print(
                    f"Row {index}: Fallback triggered, extracted {len(description_text)} characters."
                )

        except Exception as e:
            print(f"Row {index}: Connection severed or DOM timeout.")
            df.at[index, "Description"] = "Error"

        # State Persistence: Save matrix every 20 iterations to prevent data loss
        if (index + 1) % 20 == 0:
            df.to_csv("dubizzle_laptops_deep.csv", index=False)
            print("--- Memory Checkpoint Persisted ---")

    driver.quit()
    df.to_csv("dubizzle_laptops_deep.csv", index=False)
    print("\nSUCCESS! Unstructured NLP payload harvested and saved.")


if __name__ == "__main__":
    scrape_deep_descriptions()
