import csv

def read_csv(file):
    result = list()
    with open(file) as csvfile:
        # Read csv file
        reader = csv.reader(csvfile)
        # Ignore header
        next(reader, None)
        for row in reader:
            result.append(row)

    # Convert all elements into float
    new_result = [[float(x) for x in y] for y in result]

    # Convert all target values into integer

    return new_result