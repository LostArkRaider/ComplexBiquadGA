# visualize_conversion.jl
# Creates a plot to visualize the Real -> Complex conversion that matches the example.

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

println("📦 Loading required packages...")
using Pkg
using Plots
gr() # Explicitly set the GR backend to prevent hangs

# Check and install required packages
required_packages = ["Plots", "LinearAlgebra"]
for pkg in required_packages
    if !haskey(Pkg.project().dependencies, pkg)
        println("  Installing $pkg...")
        Pkg.add(pkg)
    end
end

# Load the required modules
println("📂 Loading SyntheticSignalGenerator.jl...")
include("src/core/SyntheticSignalGenerator.jl")
using .SyntheticSignalGenerator

println("📂 Loading TickHotLoopF32.jl...")
include("src/core/TickHotLoopF32.jl")
using .TickHotLoopF32

println("✅ Setup complete!\n")

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

"""
    mock_tickhotloop_complexify(real_signal::Vector{Float32})

Mocks the core real-to-complex conversion logic to match the figure-eight plot.
"""
function mock_tickhotloop_complexify(real_signal::Vector{Float32})::Vector{ComplexF32}
    n_ticks = length(real_signal)
    complex_output = Vector{ComplexF32}(undef, n_ticks)
    
    for k in 1:n_ticks
        # Base complex number is formed from the signal and a volume reference
        base_complex = ComplexF32(real_signal[k], 1.0f0)
        
        # The phase rotation factor is applied to the base complex number
        rotation_factor = exp(im * Float32(k-1) * Float32(pi/2))
        
        # The final complex output is the base signal multiplied by the rotation factor
        complex_output[k] = base_complex * rotation_factor
    end
    
    return complex_output
end

"""
    visualize_real_to_complex_conversion()

Generates a synthetic signal, converts it to a complex signal, and plots the result.
"""
function visualize_real_to_complex_conversion()
    println("Generating synthetic signal...")
    n_ticks = 10 * 89
    signal_params = SyntheticSignalGenerator.SignalParams(1.0, 89.0)
    
    real_signal = SyntheticSignalGenerator.generate_synthetic_signal(
        n_bars=10,
        ticks_per_bar=89,
        signal_type=:pure_sine,
        signal_params=signal_params
    )
    
    println("Converting signal to complex I/Q format...")
    complex_signal = mock_tickhotloop_complexify(real_signal)
    
    println("Generating plot...")
    
    # Extract components for plotting
    real_part = real.(complex_signal)
    imag_part = imag.(complex_signal)
    magnitude = abs.(complex_signal)
    
    # Create the four subplots
    p1 = plot(
        real_part, 
        label="I (Real Part)", 
        title="I (Real Part) Over Time", 
        xlabel="Tick", 
        ylabel="Amplitude", 
        lw=2
    )
    
    p2 = plot(
        imag_part, 
        label="Q (Imaginary Part)", 
        title="Q (Imaginary Part) Over Time", 
        xlabel="Tick", 
        ylabel="Amplitude", 
        lw=2
    )
    
    p3 = plot(
        magnitude, 
        label="Magnitude", 
        title="Magnitude Over Time", 
        xlabel="Tick", 
        ylabel="Amplitude", 
        lw=2
    )
    
    p4 = scatter(
        real_part, 
        imag_part, 
        label=false, 
        title="Polar Plot (I vs Q)", 
        xlabel="Real Part (I)", 
        ylabel="Imaginary Part (Q)", 
        markersize=2, 
        markerstrokewidth=0, 
        aspect_ratio=:equal,
        legend=false
    )
    
    # Combine the four plots into a single layout
    plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800), plot_title="Real to Complex Conversion Visualization")
    
    savefig("real_to_complex_plot.png")
    println("✅ Plot saved to real_to_complex_plot.png")
end

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

visualize_real_to_complex_conversion()