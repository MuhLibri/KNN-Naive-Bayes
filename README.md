# Tugas Besar 2 IF3170 Inteligensi Buatan - don't mine at night
This is a program to implement a couple of machine learning algorithms: K-Nearest Neighbors (KNN) and Naive-Bayes.

## Author

| Nama                           |   NIM    |
| ------------------------------ | :------: |
| Kevin John Wesley Hutabarat    | 13521042 |
| Muhammad Equillibrie Fajria    | 13521047 |
| M Farrel Danendra Rachim       | 13521048 |
| Jericho Russel Sebastian       | 13521107 |

## Data attributes
- 21 attributes, including numeric columns (decimal) and non-numeric columns (0s and 1s)
- A target attribute "price_range" that has a value range of 0-3.
- On ```test.csv```, there's an additional column showing the ID for each data row.

## Feature
- Calculating the accuracy, precision, and recall scores of the given training data ```data_train.csv``` and validation data ```data_validation.csv``` using KNN and Naive-Bayes (using both manual implementation and sci-kit library)
- Writing the above classification on ```output/knn_result.csv```
- Predicting the classification of the "price_range" test data ```test.csv``` and writing the result on ```output/knn_test_result_2.csv```

## Requirement

- Python 3.11 or above

## How to Execute
1. Download the source code
2. Open your preferred terminal on the root folder of the repository
3. Run the command ```python main.py``` 
4. Fill in the input according to the prompt given (type 1 or 2)
5. The program will process the data for you.