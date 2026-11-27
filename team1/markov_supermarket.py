# %%
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

try:
    from IPython.display import display
except ImportError:
    display = print

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_FOLDER = os.path.join(BASE_DIR, "data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "figures")

STATES = ["dairy", "drinks", "fruit", "spices", "checkout"]
TERMINAL_STATE = "checkout"
LARGE_N = 100

SHOW_PLOTS = True

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)


def finish_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_FOLDER, f"{name}.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# %%
def read_csv_flexible(path):
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    return df


csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER}.")

frames = []
for file in csv_files:
    daily = read_csv_flexible(file)
    daily["day"] = os.path.splitext(os.path.basename(file))[0]
    frames.append(daily)
data = pd.concat(frames, ignore_index=True)

data.columns = [str(c).strip().lower() for c in data.columns]
required_columns = {"timestamp", "customer_no", "location"}
missing_columns = required_columns - set(data.columns)
if missing_columns:
    raise ValueError(f"Required columns missing: {missing_columns}")

rows_before = len(data)
data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
data["customer_no"] = pd.to_numeric(data["customer_no"], errors="coerce")
data["location"] = data["location"].astype("string").str.strip().str.lower()
data = data.dropna(subset=["timestamp", "customer_no", "location"]).copy()
data = data[data["location"].isin(STATES)].copy()

key = ["day", "timestamp", "customer_no", "location"]
duplicate_count = data.duplicated(subset=key).sum()
data = data.drop_duplicates(subset=key).copy()

print("Rows before cleaning:", rows_before)
print("Exact duplicate records removed:", duplicate_count)
print("Rows after cleaning:", len(data))
print("Missing values after cleaning:")
print(data[["timestamp", "customer_no", "location"]].isna().sum())
print("\nDataset shape:", data.shape)
print("Unique customer IDs:", data["customer_no"].nunique())
print("Days represented:", data["day"].nunique())
print("States in data:", sorted(data["location"].unique()))
print("\nRecords by state:")
print(data["location"].value_counts().reindex(STATES, fill_value=0))
print("\nFirst records:")
display(data.head())


# %%
data = data.sort_values(["day", "customer_no", "timestamp"], kind="stable").reset_index(drop=True)

journey_series = data.groupby(["day", "customer_no"], sort=False)["location"].apply(list)


def truncate_at_first_checkout(path):
    if TERMINAL_STATE in path:
        return path[:path.index(TERMINAL_STATE) + 1]
    return path


journeys = journey_series.apply(truncate_at_first_checkout)
journeys = journeys[journeys.apply(len) > 0]

print("Customer-day journeys:", len(journeys))
print("Average recorded states per journey:", round(journeys.apply(len).mean(), 3))
print("Journey length summary:")
display(journeys.apply(len).describe().to_frame("journey_length"))


# %%
transition_records = []
for (day, customer), path in journeys.items():
    for current_state, next_state in zip(path[:-1], path[1:]):
        transition_records.append({
            "day": day,
            "customer_no": customer,
            "location": current_state,
            "next_location": next_state,
        })

transitions = pd.DataFrame(
    transition_records,
    columns=["day", "customer_no", "location", "next_location"],
)

transition_counts = (
    transitions.groupby(["location", "next_location"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=STATES, columns=STATES, fill_value=0)
)

print("Transition counts:")
display(transition_counts)

outgoing_counts = transition_counts.sum(axis=1)
P = transition_counts.div(outgoing_counts.replace(0, np.nan), axis=0)

for state in STATES:
    if outgoing_counts[state] == 0:
        P.loc[state, :] = 0.0
        P.loc[state, state] = 1.0

P.loc[TERMINAL_STATE, :] = 0.0
P.loc[TERMINAL_STATE, TERMINAL_STATE] = 1.0
P = P.reindex(index=STATES, columns=STATES).fillna(0.0)

print("\nOne-step transition matrix P:")
display(P.round(4))

print("\nRow sums (must all equal 1):")
display(P.sum(axis=1).round(8).to_frame("row_sum"))
assert np.allclose(P.sum(axis=1).to_numpy(), 1.0), "A row of P does not sum to 1."

plt.figure(figsize=(9, 7))
sns.heatmap(P, annot=True, fmt=".3f", cmap="Blues",
            vmin=0, vmax=1, linewidths=0.5, square=True)
plt.title("One-step Transition Probability Matrix P")
plt.xlabel("Next state")
plt.ylabel("Current state")
finish_plot("transition_matrix_heatmap")

plt.figure(figsize=(9, 7))
sns.heatmap(transition_counts, annot=True, fmt="d", cmap="Greens",
            linewidths=0.5, square=True)
plt.title("Observed Transition Counts")
plt.xlabel("Next state")
plt.ylabel("Current state")
finish_plot("transition_counts_heatmap")


# %%
EDGE_THRESHOLD = 0.03
G = nx.DiGraph()
G.add_nodes_from(STATES)
for s1 in STATES:
    for s2 in STATES:
        prob = float(P.loc[s1, s2])
        if prob >= EDGE_THRESHOLD:
            G.add_edge(s1, s2, probability=prob, label=f"{prob:.2f}")

pos = nx.circular_layout(G)
plt.figure(figsize=(12, 9))
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="lightblue")
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
nx.draw_networkx_edges(
    G, pos, arrows=True, arrowstyle="-|>", arrowsize=22,
    width=1.5, connectionstyle="arc3,rad=0.12",
)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=nx.get_edge_attributes(G, "label"), font_size=8
)
plt.title(f"State Transition Diagram (edges with P >= {EDGE_THRESHOLD})")
plt.axis("off")
finish_plot("state_transition_diagram")


# %%
def power_matrix(P_df, n):
    values = np.linalg.matrix_power(P_df.to_numpy(dtype=float), n)
    return pd.DataFrame(values, index=P_df.index, columns=P_df.columns)


step_numbers = [2, 3, 5, LARGE_N]
P_powers = {}

for n in step_numbers:
    P_powers[n] = power_matrix(P, n)
    print(f"\nP^{n}:")
    display(P_powers[n].round(4))

for n in step_numbers:
    plt.figure(figsize=(9, 7))
    sns.heatmap(P_powers[n], annot=True, fmt=".3f", cmap="Purples",
                vmin=0, vmax=1, linewidths=0.5, square=True)
    plt.title(f"{n}-step Transition Probability Matrix (P^{n})")
    plt.xlabel("State after n steps")
    plt.ylabel("Starting state")
    finish_plot(f"P_power_{n}_heatmap")

selected_pairs = [
    ("dairy", "checkout"),
    ("drinks", "checkout"),
    ("fruit", "checkout"),
    ("spices", "checkout"),
    ("dairy", "drinks"),
    ("drinks", "fruit"),
]

print("\nSelected n-step transition probabilities:")
for n in [2, 3, LARGE_N]:
    print(f"\n--- n = {n} ---")
    for start, end in selected_pairs:
        probability = P_powers[n].loc[start, end]
        print(f"{start} -> {end} in {n} steps: {probability:.4f}")


# %%
first_locations = journeys.apply(lambda path: path[0])
pi0 = first_locations.value_counts(normalize=True).reindex(STATES, fill_value=0.0)
pi0 = pi0 / pi0.sum()

print("Initial distribution pi_0:")
display(pi0.to_frame("probability").round(4))


def distribution_after_n(initial_distribution, transition_matrix, n):
    matrix_n = np.linalg.matrix_power(transition_matrix.to_numpy(dtype=float), n)
    values = initial_distribution.to_numpy(dtype=float) @ matrix_n
    return pd.Series(values, index=transition_matrix.columns, name=f"n={n}")


n_values_for_distribution = [0, 1, 2, 3, 5, 10, LARGE_N]
state_distributions = {}

for n in n_values_for_distribution:
    if n == 0:
        state_distributions[n] = pi0.copy()
        state_distributions[n].name = "n=0"
    else:
        state_distributions[n] = distribution_after_n(pi0, P, n)
    print(f"\nState distribution after {n} steps (pi_0 P^{n}):")
    display(state_distributions[n].to_frame().round(4))

distribution_plot = pd.DataFrame({
    f"n={n}": state_distributions[n] for n in n_values_for_distribution
})
distribution_plot.plot(kind="bar", figsize=(12, 6))
plt.title("Customer State Distribution After n Steps")
plt.xlabel("State")
plt.ylabel("Probability")
plt.xticks(rotation=0)
plt.legend(title="Number of steps")
finish_plot("state_distributions")

checkout_probability_table = pd.DataFrame({
    f"n={n}": P_powers[n][TERMINAL_STATE] for n in [2, 3, LARGE_N]
})
print("\nProbability of being in Checkout after n steps, by starting state:")
display(checkout_probability_table.round(4))


# %%
for state in STATES:
    destinations = P.loc[state].drop(labels=[TERMINAL_STATE], errors="ignore")
    if len(destinations) and destinations.sum() > 0:
        most_likely = destinations.idxmax()
        print(f"From {state}, the most likely non-checkout destination is "
              f"{most_likely} ({destinations[most_likely]:.3f}).")
    elif state == TERMINAL_STATE:
        print("Checkout is absorbing: P[checkout, checkout] = 1.")

print(f"\nProbability of being in Checkout after {LARGE_N} steps "
      f"(from the initial distribution): {state_distributions[LARGE_N][TERMINAL_STATE]:.4f}")


# %%
transition_counts.to_csv(os.path.join(OUTPUT_FOLDER, "transition_counts.csv"))
P.to_csv(os.path.join(OUTPUT_FOLDER, "transition_matrix_P.csv"))

for n, matrix in P_powers.items():
    matrix.to_csv(os.path.join(OUTPUT_FOLDER, f"P_power_{n}.csv"))

pi0.to_frame("probability").to_csv(os.path.join(OUTPUT_FOLDER, "initial_distribution_pi0.csv"))
distribution_plot.to_csv(os.path.join(OUTPUT_FOLDER, "state_distributions_after_n.csv"))
checkout_probability_table.to_csv(os.path.join(OUTPUT_FOLDER, "checkout_probabilities.csv"))
journeys.reset_index(name="journey").to_csv(
    os.path.join(OUTPUT_FOLDER, "cleaned_customer_journeys.csv"), index=False)

print("Results saved in:", OUTPUT_FOLDER)
