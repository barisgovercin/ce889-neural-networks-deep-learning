# only random library is used
import random

def cleanConvertCsv(line):  # clean rows and convert/return [x, y, vy, vx].
    line = line.strip()

    # skip empty lines
    if line == "":
        return None

    parts = line.split(",")

    # skip if column number is less than 4
    if len(parts) < 4:
        return None

    raw_x = parts[0].strip()
    raw_y = parts[1].strip()
    raw_vy = parts[2].strip()
    raw_vx = parts[3].strip()

    # skip if any one of the column is empty
    if raw_x == "" or raw_y == "" or raw_vy == "" or raw_vx == "":
        return None

    # try converting to float (skip if fails)
    try:
        x_value = float(raw_x)
        y_value = float(raw_y)
        vel_y = float(raw_vy)
        vel_x = float(raw_vx)
    except:
        return None

    return [x_value, y_value, vel_y, vel_x]


def readRawCsv(path): # reads the raw "ce889_dataCollection.csv" file and returns list of rows of [x, y, vy, vx]
    rows = []

    try:
        file_handle = open(path, "r")
    except:
        print("File could not be opened:", path)
        return rows

    # skip header
    header_line = file_handle.readline()

    # read all lines and clean the wrong ones
    for line in file_handle:
        parsed_row = cleanConvertCsv(line)
        if parsed_row is not None:
            rows.append(parsed_row)

    file_handle.close()
    return rows



def minMaxComp(values):
    if len(values) == 0:    # if list is epty, return default min-max as 0 and 1
        return 0.0, 1.0

    min_value = values[0]
    max_value = values[0]

    index = 1
    while index < len(values):
        current_value = values[index]
        if current_value < min_value:
            min_value = current_value
        if current_value > max_value:
            max_value = current_value
        index = index + 1

    # avoiding the division by zero error
    if min_value == max_value:
        max_value = min_value + 1.0

    return min_value, max_value


def normaliseValue(value, min_value, max_value):   # min-max normalisation to [0, 1].
    
    return (value - min_value) / (max_value - min_value)


def saveNormalisedCsv(path, rows, x_min, x_max, y_min, y_max):
    try:
        file_handle = open(path, "w")
    except:
        print("Cannot write to file:", path)
        return

    # header
    file_handle.write("x_target_norm,y_target_norm,new_vel_y,new_vel_x\n")

    row_index = 0
    while row_index < len(rows):
        x_value = rows[row_index][0]
        y_value = rows[row_index][1]
        vel_y = rows[row_index][2]
        vel_x = rows[row_index][3]

        x_norm = normaliseValue(x_value, x_min, x_max)
        y_norm = normaliseValue(y_value, y_min, y_max)

        line = (
            str(x_norm) + "," +
            str(y_norm) + "," +
            str(vel_y) + "," +
            str(vel_x) + "\n"
        )
        file_handle.write(line)

        row_index = row_index + 1

    file_handle.close()


def saveMinMaxValue(path, x_min, x_max, y_min, y_max):  # create scaler.py with constants: X_MIN, X_MAX, Y_MIN, Y_MAX
    try:
        file_handle = open(path, "w")
    except:
        print("Scaler module cannot written:", path)
        return
    
    file_handle.write("X_MIN = " + str(x_min) + "\n")
    file_handle.write("X_MAX = " + str(x_max) + "\n")
    file_handle.write("Y_MIN = " + str(y_min) + "\n")
    file_handle.write("Y_MAX = " + str(y_max) + "\n")

    file_handle.close()


def trainTestValSplit(rows, train_ratio, val_ratio, test_ratio):    # spliting rows into train, validation and test sets.
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0.0:
        print("Invalid split ratios.")
        return rows, [], []

    if abs(total_ratio - 1.0) > 1e-6:
        print("Warning: ratios do not sum to 1. Using them as given.")

    # make a copy of the rows
    rows_copy = list(rows)

    # shuffle the copy
    random.shuffle(rows_copy)

    total_samples = len(rows_copy)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    # test gets the remaining samples
    test_size = total_samples - train_size - val_size

    train_rows = rows_copy[0:train_size]
    val_rows = rows_copy[train_size:train_size + val_size]
    test_rows = rows_copy[train_size + val_size:]

    print("Total samples:", total_samples)
    print("Train samples:", len(train_rows))
    print("Validation samples:", len(val_rows))
    print("Test samples:", len(test_rows))

    return train_rows, val_rows, test_rows


def main():
    raw_path = "ce889_dataCollection.csv"
    normalized_path_all = "ce889_dataCollection_norm.csv"
    scaler_module_path = "scaler.py"

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    train_norm_path = "ce889_data_train_norm.csv"
    val_norm_path = "ce889_data_val_norm.csv"
    test_norm_path = "ce889_data_test_norm.csv"

    print("Reading raw data from:", raw_path)
    rows = readRawCsv(raw_path)

    if len(rows) == 0:
        print("No data, nothing to do.")
        return

    # collect x and y lists
    x_values = []
    y_values = []

    row_index = 0
    while row_index < len(rows):
        x_values.append(rows[row_index][0])
        y_values.append(rows[row_index][1])
        row_index = row_index + 1

    x_min, x_max = minMaxComp(x_values)
    y_min, y_max = minMaxComp(y_values)

    print("x_target min/max:", x_min, x_max)
    print("y_target min/max:", y_min, y_max)

    # writing normalised CSV with all the data
    print("Writing normalised csv to:", normalized_path_all)
    saveNormalisedCsv(normalized_path_all, rows, x_min, x_max, y_min, y_max)

    # spliting rows into train / val / test
    train_rows, val_rows, test_rows = trainTestValSplit(
        rows,
        train_ratio,
        val_ratio,
        test_ratio
    )

    # writing normalised CSVs for each split
    print("Writing Train normalised csv to:", train_norm_path)
    saveNormalisedCsv(train_norm_path, train_rows, x_min, x_max, y_min, y_max)

    print("Writing Validation   normalised csv to:", val_norm_path)
    saveNormalisedCsv(val_norm_path, val_rows, x_min, x_max, y_min, y_max)

    print("Writing Test  normalised csv to:", test_norm_path)
    saveNormalisedCsv(test_norm_path, test_rows, x_min, x_max, y_min, y_max)

    # writing scaler module
    print("Writing scaler module to:", scaler_module_path)
    saveMinMaxValue(scaler_module_path, x_min, x_max, y_min, y_max)

    print("Normalisation is done.")


if __name__ == "__main__":
    main()
