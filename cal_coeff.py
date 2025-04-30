import numpy as np

def fourth_degree_fit(x, y):
    # Ensure x and y are numpy arrays for computation
    x = np.array(x)
    y = np.array(y)

    # Fit a 4th-degree polynomial to the data
    coeff = np.polyfit(x, y, 4)

    return coeff

if __name__ == "__main__":
    # Example data
    x = [1,      0.9,    0.8,   0.7,    0.6,    0.5]
    y = [503.75, 520.41, 533.69, 547.11, 561.2, 584.78]
    # Get the coefficients for the 4th-degree polynomial fit
    coeff = fourth_degree_fit(x, y)

    # Display the result
    print("Fitted 4th-degree polynomial coefficients:")
    print(', '.join(map(str, coeff)))
