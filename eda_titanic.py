"""
Titanic Real Dataset EDA — generates all plots
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = "/home/claude/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/train.csv")

# ── Print stats for PDF text ──────────────────────────────────────────────────
print("SHAPE:", df.shape)
print("\nDTYPES:\n", df.dtypes)
print("\nDESCRIBE:\n", df.describe().round(2).to_string())
print("\nMISSING:\n", df.isnull().sum())
print("\nSURVIVED:\n", df["Survived"].value_counts())
print("\nPCLASS:\n", df["Pclass"].value_counts().sort_index())
print("\nSEX:\n", df["Sex"].value_counts())
print("\nEMBARKED:\n", df["Embarked"].value_counts())
surv_sex = df.groupby("Sex")["Survived"].mean().round(3)
print("\nSURV RATE BY SEX:\n", surv_sex)
surv_pclass = df.groupby("Pclass")["Survived"].mean().round(3)
print("\nSURV RATE BY PCLASS:\n", surv_pclass)

BLUE  = "#4C72B0"
ORANGE= "#DD8452"
PAL   = [ORANGE, BLUE]

# ── Plot 1: Age Histogram ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["Age"].dropna(), bins=30, color=BLUE, edgecolor="white")
ax.set_title("Age Distribution of Passengers", fontsize=14, fontweight="bold")
ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
ax.axvline(df["Age"].mean(), color="red", linestyle="--", label=f"Mean: {df['Age'].mean():.1f}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_age_histogram.png", dpi=150); plt.close()

# ── Plot 2: Fare Histogram ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["Fare"].dropna(), bins=40, color=ORANGE, edgecolor="white")
ax.set_title("Fare Distribution (Right-Skewed)", fontsize=14, fontweight="bold")
ax.set_xlabel("Fare (£)"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_fare_histogram.png", dpi=150); plt.close()

# ── Plot 3: Survival Count ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Survived"].value_counts().sort_index()
bars = ax.bar(["Did Not Survive", "Survived"], counts.values, color=PAL, edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=11)
ax.set_title("Overall Survival Count", fontsize=14, fontweight="bold")
ax.set_ylabel("Count"); ax.set_ylim(0, max(counts.values)*1.2)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_survival_count.png", dpi=150); plt.close()

# ── Plot 4: Survival by Sex ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(x="Sex", hue="Survived", data=df, palette=PAL, ax=ax)
ax.set_title("Survival by Gender", fontsize=14, fontweight="bold")
ax.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_survival_by_sex.png", dpi=150); plt.close()

# ── Plot 5: Survival by Pclass ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(x="Pclass", hue="Survived", data=df, palette=PAL, ax=ax)
ax.set_title("Survival by Passenger Class", fontsize=14, fontweight="bold")
ax.legend(title="Survived", labels=["No", "Yes"])
ax.set_xlabel("Passenger Class (1=1st, 2=2nd, 3=3rd)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_survival_by_pclass.png", dpi=150); plt.close()

# ── Plot 6: Age Boxplot vs Survived ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
survived_labels = df["Survived"].map({0: "Did Not Survive", 1: "Survived"})
df2 = df.copy(); df2["Outcome"] = survived_labels
sns.boxplot(x="Outcome", y="Age", data=df2, hue="Outcome",
            palette=PAL, ax=ax, legend=False)
ax.set_title("Age Distribution vs Survival", fontsize=14, fontweight="bold")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_age_boxplot.png", dpi=150); plt.close()

# ── Plot 7: Fare Boxplot vs Survived ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(x="Outcome", y="Fare", data=df2, hue="Outcome",
            palette=PAL, ax=ax, legend=False)
ax.set_title("Fare Distribution vs Survival", fontsize=14, fontweight="bold")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_fare_boxplot.png", dpi=150); plt.close()

# ── Plot 8: Correlation Heatmap ───────────────────────────────────────────────
num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.8, annot_kws={"size": 11}, ax=ax)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_heatmap.png", dpi=150); plt.close()

# ── Plot 9: Pairplot ──────────────────────────────────────────────────────────
pair_df = df[["Survived", "Pclass", "Age", "Fare", "SibSp"]].dropna().copy()
pair_df["Survived"] = pair_df["Survived"].astype(str)
g = sns.pairplot(pair_df, hue="Survived", palette={"0": ORANGE, "1": BLUE},
                 plot_kws={"alpha": 0.5}, height=2.2)
g.fig.suptitle("Pairplot of Key Features (hue = Survived)", y=1.01, fontsize=12)
g.savefig(f"{PLOT_DIR}/09_pairplot.png", dpi=110); plt.close()

# ── Plot 10: Survival Rate by Pclass & Sex ────────────────────────────────────
surv_rate = df.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=surv_rate,
            palette=[BLUE, ORANGE], ax=ax)
ax.set_title("Survival Rate by Passenger Class & Gender", fontsize=14, fontweight="bold")
ax.set_ylabel("Survival Rate"); ax.set_xlabel("Passenger Class")
ax.set_ylim(0, 1.1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0%}",
                (p.get_x() + p.get_width()/2, p.get_height() + 0.02),
                ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/10_surv_rate_pclass_sex.png", dpi=150); plt.close()

# ── Plot 11: Violin Age by Pclass ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
sns.violinplot(x="Pclass", y="Age", data=df, hue="Pclass",
               palette="muted", ax=ax, legend=False)
ax.set_title("Age Distribution by Passenger Class", fontsize=14, fontweight="bold")
ax.set_xlabel("Passenger Class")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/11_age_violin.png", dpi=150); plt.close()

# ── Plot 12: Embarked Survival ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(x="Embarked", hue="Survived", data=df, palette=PAL, ax=ax,
              order=["S", "C", "Q"])
ax.set_title("Survival by Embarkation Port", fontsize=14, fontweight="bold")
ax.set_xlabel("Port (S=Southampton, C=Cherbourg, Q=Queenstown)")
ax.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/12_embarked_survival.png", dpi=150); plt.close()

# ── Plot 13: Missing Values Bar ───────────────────────────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 3))
missing.plot(kind="bar", color=BLUE, ax=ax, edgecolor="white", rot=0)
for i, v in enumerate(missing):
    ax.text(i, v + 1, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontsize=10)
ax.set_title("Missing Values per Column", fontsize=13, fontweight="bold")
ax.set_ylabel("Count"); ax.set_ylim(0, missing.max() * 1.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/13_missing_values.png", dpi=150); plt.close()

print("\nAll 13 plots saved to", PLOT_DIR)
