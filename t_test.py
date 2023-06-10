import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def t_test(sample, mu):
    n = len(sample)
    df = n - 1
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    t = (mean - mu) / (std / np.sqrt(n))
    p_value = stats.t.sf(np.abs(t), df) * 2
    return t, p_value

# Streamlit app
def main():
    st.title("T-Distribution Test")
    
    # Input data
    st.header("Input Data")
    sample_size = st.number_input("Sample Size", min_value=1, step=1, value=30)
    sample_mean = st.number_input("Sample Mean")
    population_mean = st.number_input("Population Mean")
    confidence_level = st.slider("Confidence Level", min_value=0.01, max_value=0.99, step=0.01, value=0.95)
    
    # Generate sample data
    np.random.seed(42)
    sample_data = np.random.randn(sample_size) + sample_mean
    
    # Perform t-test
    t_statistic, p_value = t_test(sample_data, population_mean)
    
    # Display results
    st.header("Test Results")
    st.write(f"Sample Size: {sample_size}")
    st.write(f"Sample Mean: {sample_mean}")
    st.write(f"Population Mean: {population_mean}")
    st.write(f"Confidence Level: {confidence_level}")
    st.write(f"T-Statistic: {t_statistic}")
    st.write(f"P-Value: {p_value}")
    
    # Plotting
    st.header("Distribution Plot")
    x = np.linspace(stats.t.ppf(0.001, sample_size - 1), stats.t.ppf(0.999, sample_size - 1), 1000)
    y = stats.t.pdf(x, sample_size - 1)
    
    plt.plot(x, y, label=f"df = {sample_size - 1}")
    plt.fill_between(x, y, where=(x >= stats.t.ppf(0.001, sample_size - 1)) & (x <= stats.t.ppf(0.999, sample_size - 1)), alpha=0.2)
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.legend()
    
    st.pyplot(plt)
    
if __name__ == "__main__":
    main()
