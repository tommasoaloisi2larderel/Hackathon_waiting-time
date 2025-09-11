import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/waiting_times_train.csv", parse_dates=["DATETIME"])

# Extract the hour of the day
df["HOUR"] = df["DATETIME"].dt.hour
df = df[df["DATETIME"].dt.date == pd.to_datetime("2022-07-05").date()]

plt.figure(figsize=(10, 6))

# Create a 2D histogram to count points
hb = plt.hexbin(df["HOUR"], df["CURRENT_WAIT_TIME"], gridsize=24, cmap="viridis", mincnt=1)

# Add a color bar to show density
cb = plt.colorbar(hb)
cb.set_label("Number of observations")

# Customize the plot
plt.title("Current Wait Time vs Hour of Day (Density Colored)")
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Current Wait Time (minutes)")
plt.grid(True)
plt.xticks(range(0, 24))

plt.show()

plt.figure(figsize=(10, 6))

# Create a 2D histogram to count points
hb = plt.hexbin(df["CURRENT_WAIT_TIME"], df["WAIT_TIME_IN_2H"], gridsize=30, cmap="plasma", mincnt=1)

# Add a color bar to show density
cb = plt.colorbar(hb)
cb.set_label("Number of observations")

# Customize the plot
plt.title("Wait Time in 2 Hours vs Current Wait Time (Density Colored)")
plt.xlabel("Current Wait Time (minutes)")
plt.ylabel("Wait Time in 2 Hours (minutes)")
plt.grid(True)

plt.show()