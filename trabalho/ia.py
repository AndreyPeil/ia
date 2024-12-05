
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Equipment Needed'] = df['Equipment Needed'].replace('None', 0)
    label_encoder = LabelEncoder()
    df['Benefit'] = label_encoder.fit_transform(df['Benefit'])
    df['Target Muscle Group'] = label_encoder.fit_transform(df['Target Muscle Group'])
    df['Difficulty Level'] = label_encoder.fit_transform(df['Difficulty Level'])
    X = df.drop(columns=['Name of Exercise', 'Burns Calories (per 30 min)'])
    y = df['Burns Calories (per 30 min)']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    y = y.values.astype(np.float32)
    return df, X, y


def build_and_train_model(X_train, y_train, input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=1000, batch_size=5, verbose=0)
    return model


def generate_week_plan(df, predictions, days=5, exercises_per_day=5):
    df['Predicted Calories Burned'] = predictions
    df_sorted = df.sort_values(by='Predicted Calories Burned', ascending=False)
    all_exercises = df_sorted[['Name of Exercise', 'Target Muscle Group', 'Sets', 'Reps',
                               'Predicted Calories Burned', 'Difficulty Level']]
    week_plan = []
    used_exercises = set()

    for _ in range(days):
        daily_plan = []
        for _, exercise_row in all_exercises.iterrows():
            if len(daily_plan) == exercises_per_day:
                break
            if exercise_row['Name of Exercise'] not in used_exercises:
                daily_plan.append(exercise_row)
                used_exercises.add(exercise_row['Name of Exercise'])
        week_plan.append(daily_plan)

    return week_plan, df_sorted


def display_week_plan(week_plan):
    print("\nPlano de Exercícios para a Semana:")
    for day, exercises in enumerate(week_plan, start=1):
        print(f"\nDia {day}:")
        for exercise in exercises:
            print(f"  Exercício: {exercise['Name of Exercise']}")
            print(f"    Grupo Muscular: {exercise['Target Muscle Group']}")
            print(f"    Séries: {exercise['Sets']}, Repetições: {exercise['Reps']}")
            print(f"    Calorias Queimadas (predição): {exercise['Predicted Calories Burned']:.2f}")
            print(f"    Nível de Dificuldade: {exercise['Difficulty Level']}")


def plot_calories_burned(week_plan):
    days = [f"Dia {i+1}" for i in range(len(week_plan))]
    calories_per_day = [sum([exercise['Predicted Calories Burned'] for exercise in day]) for day in week_plan]
    plt.bar(days, calories_per_day)
    plt.xlabel('Dias')
    plt.ylabel('Calorias queimadas')
    plt.title('Calorias queimadas por dia')
    plt.show()


def main_menu():
    print("Bem-vindo ao Sistema de Planejamento de Exercícios!")
    file_path =  r"C:\Users\andei\Downloads\Top 50 Excerice for your body.csv"

    df, X, y = load_and_prepare_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTreinando o modelo... Aguarde!")
    model = build_and_train_model(X_train, y_train, X_train.shape[1])

    print("\nGerando previsões...")
    predictions = model.predict(X)

    print("\nCriando plano de treino semanal...")
    week_plan, df_sorted = generate_week_plan(df, predictions)

    while True:
        print("\nMenu:")
        print("1. Ver plano de treino semanal")
        print("2. Exibir exercícios com maior queima calórica")
        print("3. Exibir gráfico de calorias queimadas por dia")
        print("4. Sair")
        choice = input("Escolha uma opção: ")

        if choice == '1':
            display_week_plan(week_plan)
        elif choice == '2':
            print("\nTop 5 exercícios com maior queima calórica:")
            print(df_sorted[['Name of Exercise', 'Predicted Calories Burned']].head())
        elif choice == '3':
            plot_calories_burned(week_plan)
        elif choice == '4':
            print("Saindo... Até a próxima!")
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main_menu()
