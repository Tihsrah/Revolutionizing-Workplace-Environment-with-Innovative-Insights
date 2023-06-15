import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")

# Define a function to calculate constructive_criticism for each row
def calculate_constructive_criticism(row):
    if row['sentiment'] == 'NEGATIVE' and row['hate'] > row['non_hate']:
        return 0
    elif row['sentiment'] == 'NEGATIVE' and row['non_hate'] > row['hate']:
        return 1
    else:
        return 0
def calculate_criticism(row):
    if row['sentiment'] == 'NEGATIVE' and row['hate'] > row['non_hate']:
        return 1
    elif row['sentiment'] == 'POSITIVE' and row['hate'] > row['non_hate']:
        return 1
    elif row['sentiment'] == 'POSITIVE' and row['non_hate'] > row['hate']:
        return 0
    else:
        return 0
# Apply the function to each row and assign the result to the 'constructive_criticism' column
df['constructive_criticism'] = df.apply(calculate_constructive_criticism, axis=1)
df['criticism']=df.apply(calculate_criticism, axis=1)

def draw_criticism_graph(df):
    # Get the row numbers
    x = df.index

    # Get the values for the 'constructive_criticism' and 'criticism' columns
    y_constructive = df['constructive_criticism']
    y_criticism = df['criticism']

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the lines
    ax.plot(x, y_constructive, marker='o', label='Constructive Criticism')
    ax.plot(x, y_criticism, marker='o', label='Criticism')

    # Add labels and title
    ax.set_xlabel('Row Number')
    ax.set_ylabel('Value')
    ax.set_title('Constructive Criticism and Criticism Line Plot')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()
draw_criticism_graph(df)