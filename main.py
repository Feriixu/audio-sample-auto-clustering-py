import tkinter as tk
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import sounddevice as sd
import os
import soundfile as sf
import hashlib
import json

# Path to your folder containing audio files
folder_path = '/home/elias/Music/samples'

# Define cache directories
samples_cache_dir = os.path.expanduser('~/.cache/audio_clustering/samples/')
features_cache_dir = os.path.expanduser('~/.cache/audio_clustering/features/')
os.makedirs(samples_cache_dir, exist_ok=True)
os.makedirs(features_cache_dir, exist_ok=True)

# Reading and processing audio files
features = []
file_names = []
file_paths = []  # Store full paths for playback

# Traverse the directory tree
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.mp3') or file.endswith('.wav'):  # Add other formats as needed
            file_path = os.path.join(root, file)
            hash_object = hashlib.sha1(file_path.encode())
            hex_dig = hash_object.hexdigest()
            cached_features_path = os.path.join(features_cache_dir, f"{hex_dig}.json")

            if os.path.exists(cached_features_path):
                # Load cached features
                with open(cached_features_path, 'r') as f:
                    mfcc_mean = json.load(f)
            else:
                try:
                    # Calculate MFCC features
                    audio, sr = librosa.load(file_path)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
                    mfcc_mean = np.mean(mfcc, axis=1).tolist()  # Convert to list for JSON serialization
                    # Cache the features
                    with open(cached_features_path, 'w') as f:
                        json.dump(mfcc_mean, f)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

            features.append(mfcc_mean)
            file_names.append(file)
            file_paths.append(file_path)

from sklearn.manifold import TSNE

# Proceed with clustering if features were extracted
if features:
    features = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    n_clusters = 15  # You may need to adjust this based on your specific needs
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)

    clusters = {}
    for file, cluster_id in zip(file_names, labels):
        clusters.setdefault(cluster_id, []).append(file)

    for cluster_id, files in clusters.items():
        print(f"Cluster {cluster_id}:")
        for file in files:
            print(f" - {file}")
else:
    print("No features extracted.")
    exit(-1)

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=3000, random_state=42)
reduced_features_tsne = tsne.fit_transform(scaled_features)

# # PCA to reduce our features to 2 dimensions for visualization
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(scaled_features)

last_played_index = -1  # Variable to keep track of the last played sample
highlighted_point = None  # Keep track of the highlighted point on the plot

# Function to play the selected audio file


# Global variable to keep track of whether a sound is currently playing
is_playing = False


def play_audio(file_path):
    global is_playing

    # Use the file path to generate a unique name for the cached file
    # This example uses SHA1 to hash the original file path
    hash_object = hashlib.sha1(file_path.encode())
    hex_dig = hash_object.hexdigest()
    cached_file_path = os.path.join(samples_cache_dir, f"{hex_dig}.wav")

    # Check if a cached version already exists
    if not os.path.exists(cached_file_path):
        # If not, convert and cache the file
        audio = AudioSegment.from_file(file_path)
        audio.export(cached_file_path, format="wav")

    # Stop any currently playing sound
    if is_playing:
        sd.stop()
        is_playing = False

    # Load the cached WAV file for playback
    data, fs = sf.read(cached_file_path, dtype='float32')
    sd.play(data, samplerate=fs, blocking=False)
    is_playing = True


def select_and_play_point(xdata, ydata, play_same_again: bool):
    global last_played_index, highlighted_point
    if xdata is not None and ydata is not None:
        # Find the closest point
        point = [xdata, ydata]
        distances = distance.cdist([point], reduced_features_tsne, 'euclidean')
        min_index = np.argmin(distances)
        # Play the audio sample only if it's newly selected
        if play_same_again or min_index != last_played_index:
            last_played_index = min_index
            selected_file_path = file_paths[min_index]
            print(f"Playing: {selected_file_path}")
            play_audio(selected_file_path)

            # Highlight the selected point in its cluster color but with distinct styling
            if highlighted_point:
                highlighted_point.remove()  # Remove the previous highlight
            cluster_color = colors[labels[min_index]]
            highlighted_point = ax.scatter(reduced_features_tsne[min_index, 0], reduced_features_tsne[min_index, 1],
                                           color=cluster_color, s=200, edgecolors='black', linewidth=2, zorder=5)
            canvas.draw()



# Function to handle mouse motion event
def onmotion(event):
    if event.button == 1:  # Check if left mouse button is pressed
        select_and_play_point(event.xdata, event.ydata, False)

def onclick(event):
    select_and_play_point(event.xdata, event.ydata, True)



# Assuming 'reduced_features', 'file_names', 'folder_path', and 'labels' are already defined and filled from your clustering script

# Set up the tkinter GUI
root = tk.Tk()
root.wm_title("Audio Samples Clustering")

# Set up the matplotlib figure and axes
fig, ax = plt.subplots()

# Set the background color of the figure and axes
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Adjusting text, ticks, and labels color to white for visibility
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.spines['top'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

# Adjust based on the number of clusters, ensure there's a color for each cluster
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'olive', 'lime', 'teal', 'cyan']

# Update this part to use `reduced_features_tsne` for plotting
for i in range(n_clusters):
    cluster_points = reduced_features_tsne[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}', alpha=0.6,
               edgecolors='w')

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Connect the event for mouse motion with button pressed
fig.canvas.mpl_connect('motion_notify_event', onmotion)
fig.canvas.mpl_connect('button_press_event', onclick)


tk.mainloop()
