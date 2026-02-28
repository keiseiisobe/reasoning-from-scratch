import matplotlib.pyplot as plt
import numpy as np

def create_visualization():
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Neural Network Inference ---
    # Concept: Fixed Model, Fixed Parameters, Direct Prediction
    x = np.linspace(0, 10, 100)
    # A fixed, learned function (e.g., a sigmoid-like curve)
    y_fixed = 1 / (1 + np.exp(-(x - 5))) 
    
    ax1.plot(x, y_fixed, color='blue', linewidth=3, label='Fixed Model f(θ)')
    
    # Example input/output
    input_x = 7.0
    output_y = 1 / (1 + np.exp(-(input_x - 5)))
    
    ax1.scatter([input_x], [output_y], color='red', s=100, zorder=5)
    ax1.annotate("Input (x)\n'Prompt'", (input_x, 0.1), xytext=(input_x, -0.15), 
                 arrowprops=dict(arrowstyle='->'), ha='center')
    ax1.annotate("Output (y)\n'Prediction'", (input_x, output_y), xytext=(input_x+0.5, output_y+0.1), 
                 arrowprops=dict(arrowstyle='->'))

    ax1.set_title("Neural Network Inference\n(Applying a Fixed Function)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Input Space")
    ax1.set_ylabel("Probability / Output")
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    ax1.text(1, 0.8, "Parameters θ are FIXED.\nNo learning happens here.", 
             bbox=dict(facecolor='white', alpha=0.8))

    # --- Plot 2: Statistical Inference ---
    # Concept: Learning/Estimating unknown parameters from data
    np.random.seed(42)
    data_x = np.linspace(2, 8, 15)
    data_y = 0.1 * data_x + 0.2 + np.random.normal(0, 0.1, len(data_x))
    
    ax2.scatter(data_x, data_y, color='black', label='Observed Data')
    
    # Show multiple possible "fits" to represent uncertainty/estimation
    x_range = np.linspace(0, 10, 100)
    ax2.plot(x_range, 0.11 * x_range + 0.18, color='green', alpha=0.4, linestyle='--')
    ax2.plot(x_range, 0.09 * x_range + 0.22, color='green', alpha=0.4, linestyle='--')
    ax2.plot(x_range, 0.1 * x_range + 0.2, color='green', linewidth=2, label='Estimated Relationship')
    
    # Confidence interval (shading)
    ax2.fill_between(x_range, 0.08 * x_range + 0.15, 0.12 * x_range + 0.25, 
                     color='green', alpha=0.1, label='Uncertainty (95% CI)')

    ax2.set_title("Statistical Inference\n(Estimating Unknown Information)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Sample Data")
    ax2.set_ylabel("Target Variable")
    ax2.set_ylim(-0.2, 1.5)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.text(0.5, 1.2, "Goal: Learn θ from data.\nQuantify uncertainty.", 
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    # Save the plot
    output_path = 'ch02/workspace/inference_terminology/inference_comparison.png'
    plt.savefig(output_path)
    print(f"Graph saved to: {output_path}")

if __name__ == "__main__":
    create_visualization()
