import argparse
import matplotlib.pyplot as plt

def find_line_numbers(filename):
    line_numbers = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f, start=1):
            if ("NavEKF2_core * this" in line or "NavEKF3_core * this" in line) and ("correct" in line.lower() or "update" in line.lower() or "predict" in line.lower() or "fuse" in line.lower()):
                line_numbers.append(i)
                print(line)
    return line_numbers

def plot_line_numbers(line_numbers, total_lines):
    plt.boxplot(line_numbers, vert=False)
    plt.title(f"Box and Whisker Plot of EKF Ranks")
    plt.xlabel("Rank")
    plt.xlim(0, total_lines)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():

    with open("closest_functions.txt", 'r') as f:
        lines = f.readlines()

    line_numbers = find_line_numbers("closest_functions.txt")
    total_lines = len(lines)
    print(len(line_numbers))

    plot_line_numbers(line_numbers, total_lines)

if __name__ == "__main__":
    main()
