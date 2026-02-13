# AudioAnalitycs

## Project Setup

Due to the size of the multiple dataset, they have been stored out of the project on this link :
https://drive.google.com/drive/folders/1k-HPTUzPT_xZc8-DYe7EUWvLc5btsngj?usp=drive_link

When the different dataset are downloaded, put them in the folder "data" in the root of the project.

You then need to download the different depedencies of this projet with this following command :
```
pip install -r requirements.txt
```

## Predict the popularity of a spotify song

### How to use it

1. Load the dataset as explained in the Project Setup section.

2. To train the model, run estim_popularite_son.py. This will train the model and save it.

3. To evaluate the model, run load_model.py. This script displays the RMSE, R² score, and two graphs to assess the model’s accuracy and visualize the popularity distribution.

### How does it work

This feature aims to estimate the popularity score of a Spotify song (from 0 to 100) based on its metadata characteristics.
The prediction is performed using a machine learning regression model trained on historical Spotify chart data.

The prediction pipeline is composed of the following steps:

1. Data loading

    * The dataset is loaded using a dedicated EstimDataLoader class.

    * Text and numerical features are identified automatically.

2. Feature selection

    * Highly correlated or irrelevant features are removed based on a correlation matrix analysis.

    * This helps reduce noise and overfitting.

3. Preprocessing

    * Numerical features are scaled using StandardScaler.

    * Textual features are encoded using an ordinal encoding strategy.

    * All preprocessing steps are encapsulated in an object-oriented preprocessor class.

4. Model training

    * A Linear Regression model is used as a baseline.

    * The model learns a linear relationship between features and popularity.

5. Model persistence

    * The trained model can be saved and reloaded to avoid retraining.

    * This enables fast inference and reproducibility.


During the training phase, several feature selection steps were applied.
To limit the dimensionality of the dataset, multiple tests were conducted and only a subset of the most relevant features was kept for model training.

### Model evaluation

The model is evaluated using standard regression metrics:

* RMSE (Root Mean Squared Error) measures the average prediction error in popularity points.

* R² score indicates how much of the variance in popularity is explained by the model.

RMSE ≈ 14
and
R² ≈ 0.12

These results show that:

* The model captures general trends in popularity but cannot show precise prediction as it's RMSE is still quite wide when you look at the distribution of the popularity. (cf load_model.py second graph)

* Extreme popularity values are harder to predict due to dataset imbalance, a lot of the popularity range from 40 to 100 while a little minority are around 0 (cf load_model.py second graph)

### Limitation
The dataset used in this project is inherently biased, as it focuses exclusively on top-charting songs across different countries. As a result, it does not fully reflect the diversity and complexity of the music industry.
Key external factors such as media exposure, marketing strategies, social influence, and promotional campaigns are not present in the dataset, which limits the accuracy of the popularity predictions.

Moreover, the linear regression model used in this project is limited in its ability to model complex and non-linear relationships, which are likely to play a significant role in determining a song’s popularity.

## Discover Music Sub-Genres
### How to use it
1. Load the dataset as explained in the Project Setup section.
2. Configure the analysis parameters in `main.py`:
```python
   DATA_FILE = 'Dataset_with_Genres_modified.csv'
   GENRE = 'rock'        # Genre to analyze
   N_CLUSTERS = 5        # Number of clusters
```
3. Run the clustering analysis:
```bash
   python main.py
```
4. The script will generate:
   - A CSV file with cluster assignments (`rock_clusters.csv`)
   - t-SNE visualization (`rock_tsne.png`)
   - Feature distributions (`rock_distributions.png`)
   - Correlation matrix (`rock_correlation.png`)

### How does it work
This feature aims to discover musical sub-genres within a main genre by clustering songs based on their audio characteristics.
The discovery is performed using an unsupervised machine learning approach that groups similar songs together.

The clustering pipeline is composed of the following steps:

1. **Data loading and preprocessing**
   - The dataset is loaded using the `DataProcessor` class.
   - Genre labels are parsed and converted to Python lists for filtering.
   - Songs are filtered by the target genre (e.g., rock, pop, rap).

2. **Feature extraction**
   - 11 audio features are used for clustering:
     * `danceability`, `energy`, `loudness`
     * `acousticness`, `instrumentalness`, `liveness`
     * `valence`, `tempo`, `speechiness`
     * `key`, `mode`
   - These features capture the sonic characteristics of each song.

3. **Data normalization**
   - Audio features are standardized using `StandardScaler` (mean=0, std=1).
   - This ensures all features contribute equally to the clustering algorithm.

4. **Clustering**
   - K-Means algorithm is applied to discover sub-genre clusters.
   - Multiple values of K are tested to find the optimal number of clusters.
   - The best K is selected based on the Silhouette Score.

5. **Cluster analysis**
   - Each cluster's profile is analyzed by computing mean feature values.
   - The dominant sub-genres in each cluster are identified.
   - The most popular songs in each cluster are displayed.

All components are implemented using **Object-Oriented Programming (OOP)**:
- `DataProcessor`: Handles data loading and filtering
- `MusicClusterer`: Performs clustering and analysis
- `ClusterVisualizer`: Generates visualizations

### Model evaluation
The clustering quality is evaluated using standard metrics:

* **Silhouette Score** measures how well-separated the clusters are (range: -1 to 1, higher is better).
* **Davies-Bouldin Index** measures cluster compactness and separation (lower is better).

For the Rock genre analysis:
- **Silhouette Score ≈ 0.139** (acceptable separation)
- **5 distinct clusters** discovered
- **74,324 songs** analyzed

These results show that:
* The algorithm successfully identifies distinct sonic patterns within the rock genre.
* Each cluster represents a coherent sub-genre with specific audio characteristics.
* Popular examples help validate that clusters correspond to recognizable sub-genres (e.g., Hard Rock, Acoustic Rock, Psychedelic Rock).

### Cluster interpretation
Example clusters discovered for Rock music:
- **Cluster 0**: Psychedelic/Spoken Rock (high speechiness)
- **Cluster 1**: Mainstream Rock (high valence, popular)
- **Cluster 2**: Electric Rock (high energy, minor mode)
- **Cluster 3**: Acoustic/Folk Rock (high acousticness)
- **Cluster 4**: Hard/Fast Rock (high tempo, high energy)

### Visualization
The project generates several visualizations to interpret the clusters:

* **t-SNE 2D projection**: Shows cluster separation in reduced dimensionality space
* **Feature distributions**: Histograms of each audio feature across all songs
* **Correlation matrix**: Identifies relationships between audio features

### Limitations
The clustering approach has several limitations:

* **Genre label dependency**: The initial filtering relies on Spotify's genre labels, which may be incomplete or inconsistent.
* **Audio features only**: The analysis uses only sonic characteristics and does not consider lyrics, cultural context, or temporal trends.
* **K-Means constraints**: The algorithm assumes spherical clusters and may not capture complex, non-convex sub-genre boundaries.
* **Label validation**: Approximately 50% of songs in the dataset have no genre labels, limiting validation against ground truth.
* **Subjective interpretation**: Musical sub-genres are culturally defined and may not always align with purely acoustic similarity.

Despite these limitations, the unsupervised approach successfully discovers meaningful musical patterns and provides interpretable clusters that correspond to recognizable sub-genres.

