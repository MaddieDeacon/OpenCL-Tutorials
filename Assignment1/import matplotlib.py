#This is just some graphs for report

import matplotlib.pyplot as plt

configurations = ['Hillis-Steele\n(1025 bins)', 'Blelloch\n(1025 bins)', 'Hillis-Steele\n(10 bins)', 'Blelloch\n(10 bins)']
times = [0.019144, 0.0174948, 0.0167292, 0.0168807]
colors = ['blue', 'red', 'green', 'purple']

#bar chart dimensions
plt.figure(figsize=(5, 3))
bars = plt.bar(configurations, times, color=colors, width=0.6)

#labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0001, f'{yval:.6f} s', ha='center', va='bottom', fontsize=8)


plt.title('Total Execution Time for 16-bit Colour Image\n(3 Channels)', fontsize=20)
plt.xlabel('Scan Variant and Bin Size', fontsize=20)
plt.ylabel('Total Execution Time (s)', fontsize=20)
plt.ylim(0.015, 0.020)
plt.yticks([0.015, 0.016, 0.017, 0.018, 0.019, 0.020], fontsize=20) #so i can see it 
plt.xticks(fontsize=20)


plt.tight_layout()
plt.savefig('scan_variant_bin_comparison.png', dpi=150)
plt.show()



#Figure 2
image_types = ['Colour (3 Channels)', 'Grayscale (1 Channel)']
times = [0.0168732, 0.005558496]
colors = ['orange', 'gray']

# Create bar chart
plt.figure(figsize=(4, 3))
bars = plt.bar(image_types, times, color=colors, width=0.5)

# Add labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0001, f'{yval:.6f} s', ha='center', va='bottom', fontsize=8)

# Customize
plt.title('Total Execution Time for 16-bit Image\n(Blelloch, 256 Bins)', fontsize=20)
plt.xlabel('Image Type', fontsize=20)
plt.ylabel('Total Execution Time (s)', fontsize=20)
plt.ylim(0.0, 0.02)
plt.yticks([0.0, 0.005, 0.010, 0.015, 0.020], fontsize=20)
plt.xticks(fontsize=20)

# Save
plt.tight_layout()
plt.savefig('rgb_vs_grayscale_256bins_comparison.png', dpi=150)
plt.show()