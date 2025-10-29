# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # =========================
# # Load Data
# # =========================
# df_qwen = pd.read_excel(
#     "/Users/ravi/Documents/Multi_Agent_RAG/Contextual_RAG_Chatbot/data/artifacts/qwen3_1_7B_ragas_evaluation_report.xlsx"
# )
# df_llama = pd.read_excel(
#     "/Users/ravi/Documents/Multi_Agent_RAG/Contextual_RAG_Chatbot/data/artifacts/llama3_8B_ragas_evaluation_report.xlsx"
# )

# # Tag model
# df_qwen["Model"] = "Qwen3-1.7B"
# df_llama["Model"] = "Llama3-8B"

# # Combine
# df = pd.concat([df_qwen, df_llama], ignore_index=True)

# # =========================
# # Select numeric metrics
# # =========================
# metrics = ["context_recall", "faithfulness", "factual_correctness(mode=f1)"]

# # Calculate mean scores per model
# df_mean = df.groupby("Model")[metrics].mean().reset_index()

# # Melt for seaborn
# df_melted = df_mean.melt(id_vars="Model", var_name="Metric", value_name="Score")

# # =========================
# # Plot Bar Chart
# # =========================
# plt.figure(figsize=(8, 6))
# sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="Set2")
# plt.title("Model Comparison Across Metrics", fontsize=14, weight="bold")
# plt.ylabel("Average Score")
# plt.xlabel("Metric")
# plt.ylim(0, 1)  # since these are evaluation metrics (0-1 scale)
# plt.legend(title="Model")
# plt.tight_layout()
# plt.savefig("model_comparison_bar.png", dpi=300, bbox_inches="tight")
# plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # =========================
# # Load Data
# # =========================
# df_qwen = pd.read_excel(
#     "/Users/ravi/Documents/Multi_Agent_RAG/Contextual_RAG_Chatbot/data/artifacts/qwen3_1_7B_ragas_evaluation_report.xlsx"
# )
# df_llama = pd.read_excel(
#     "/Users/ravi/Documents/Multi_Agent_RAG/Contextual_RAG_Chatbot/data/artifacts/llama3_8B_ragas_evaluation_report.xlsx"
# )

# # Tag model
# df_qwen["Model"] = "Qwen3-1.7B"
# df_llama["Model"] = "Llama3-8B"

# # Combine
# df = pd.concat([df_qwen, df_llama], ignore_index=True)

# # =========================
# # Select numeric metrics
# # =========================
# metrics = ["context_recall", "faithfulness", "factual_correctness(mode=f1)"]

# # =========================
# # Plot 1: Average Comparison (bar plot)
# # =========================
# df_mean = df.groupby("Model")[metrics].mean().reset_index()
# df_melted = df_mean.melt(id_vars="Model", var_name="Metric", value_name="Score")

# plt.figure(figsize=(8, 6))
# sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="Set2")
# plt.title("Model Comparison Across Metrics", fontsize=14, weight="bold")
# plt.ylabel("Average Score")
# plt.xlabel("Metric")
# plt.ylim(0, 1)
# plt.legend(title="Model")
# plt.tight_layout()
# plt.savefig("model_comparison_bar.png", dpi=300, bbox_inches="tight")
# plt.close()

# # =========================
# # Plot 2: Count of rows > 0.7 per metric per model
# # =========================
# df_count = (
#     df.groupby("Model")[metrics]
#     .apply(lambda x: (x > 0.9).sum())
#     .reset_index()
# )

# df_count_melted = df_count.melt(id_vars="Model", var_name="Metric", value_name="Count")

# plt.figure(figsize=(8, 6))
# sns.barplot(data=df_count_melted, x="Metric", y="Count", hue="Model", palette="Set1")
# plt.title("Count of Rows with Score > 0.9", fontsize=14, weight="bold")
# plt.ylabel("Number of Rows")
# plt.xlabel("Metric")
# plt.legend(title="Model")
# plt.tight_layout()
# plt.savefig("rows_above_0.9.png", dpi=300, bbox_inches="tight")
# plt.close()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load Data
# =========================
df_qwen = pd.read_excel(
    "/Users/ravi/Documents/Multi_Agent_RAG/Contextual_RAG_Chatbot/data/artifacts/qwen3_1_7B_ragas_evaluation_report.xlsx"
)
df_llama = pd.read_excel(
    "/Users/ravi/Documents/Multi_Agent_RAG/Contextual_RAG_Chatbot/data/artifacts/llama3_8B_ragas_evaluation_report.xlsx"
)

# Tag model
df_qwen["Model"] = "Qwen3-1.7B"
df_llama["Model"] = "Llama3-8B"

# Combine
df = pd.concat([df_qwen, df_llama], ignore_index=True)

# =========================
# Select numeric metrics
# =========================
metrics = ["context_recall", "faithfulness", "factual_correctness(mode=f1)"]

# =========================
# Plot 1: Average Comparison (bar plot)
# =========================
df_mean = df.groupby("Model")[metrics].mean().reset_index()
df_melted = df_mean.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(8, 6))
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="Set2")
plt.title("Model Comparison Across Metrics", fontsize=14, weight="bold")
plt.ylabel("Average Score")
plt.xlabel("Metric")
plt.ylim(0, 1)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Plot 2: Count of rows > 0.7 per metric per model
# =========================
df_count = (
    df.groupby("Model")[metrics]
    .apply(lambda x: (x > 0.7).sum())
    .reset_index()
)
df_count_melted = df_count.melt(id_vars="Model", var_name="Metric", value_name="Count")

plt.figure(figsize=(8, 6))
sns.barplot(data=df_count_melted, x="Metric", y="Count", hue="Model", palette="Set1")
plt.title("Count of Rows with Score > 0.7", fontsize=14, weight="bold")
plt.ylabel("Number of Rows")
plt.xlabel("Metric")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("rows_above_0.7.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Plot 3: Count of rows where ALL metrics > 0.8
# =========================
df_all_good = (
    df.groupby("Model")
    .apply(lambda x: ((x[metrics] > 0.8).all(axis=1)).sum())
    .reset_index(name="Count")
)

plt.figure(figsize=(6, 6))
sns.barplot(data=df_all_good, x="Model", y="Count", palette="coolwarm")
plt.title("Count of Rows with ALL Metrics > 0.8", fontsize=14, weight="bold")
plt.ylabel("Number of Rows")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("rows_all_above_0.8.png", dpi=300, bbox_inches="tight")
plt.close()
