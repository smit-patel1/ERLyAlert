import pandas as pd

def preprocess_er_visits_from_new_data(input_path, output_path):
    df = pd.read_csv(input_path)

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['month'] = df['arrivalmonth'].map(month_map)

    day_map = {
        'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4,
        'Thursday': 5, 'Friday': 6, 'Saturday': 7
    }
    df['day'] = df['arrivalday'].map(day_map)

    df = df.dropna(subset=['month', 'day'])

    df['year'] = 2023
    df['ds'] = pd.to_datetime(df[['year', 'month', 'day']])

    daily_counts = df.groupby('ds').size().reset_index(name='y')
    daily_counts = daily_counts.sort_values('ds')

    daily_counts.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_er_visits_from_new_data(
        "data\converted_er_data.csv",
        "data\processed_er_data.csv"
    )
