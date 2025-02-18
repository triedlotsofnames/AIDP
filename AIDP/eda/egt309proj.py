import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from flask import Flask, render_template
import os

app = Flask(__name__)

titanic = pd.read_csv("data/train.csv")

"""# **EDA**"""

titanic.head()

titanic.info()

titanic.describe(include='all')

"""1, Missing values"""


def missing_value():
    sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
    img_path = os.path.join("static", "missing_values.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


"""2. Data analysis"""


def age_histogram():
    sns.histplot(titanic['Age'], kde=True)
    img_path = os.path.join("static", "age_histogram.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def fare_boxplot():
    sns.boxplot(data=titanic[['Fare']])
    img_path = os.path.join("static", "fare_boxplot.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def pclass_count():
    sns.countplot(x='Pclass', data=titanic)
    img_path = os.path.join("static", "Pclass_count.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def survival_by_age():
    sns.boxplot(x='Survived', y='Age', data=titanic)
    img_path = os.path.join("static", "survival_by_age.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def sex_survival():
    pd.crosstab(titanic['Sex'], titanic['Survived']).plot(kind='bar', stacked=True)
    img_path = os.path.join("static", "sex_survival.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def pairplot():
    sns.pairplot(titanic[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
    img_path = os.path.join("static", "pairplot.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def correlation_heatmap():
    numeric_df = titanic.select_dtypes(exclude='object')
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
    img_path = os.path.join("static", "correlation_heatmap.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


"""3. Outlier Detection"""


def pclass_vs_fare():
    sns.boxplot(x='Pclass', y='Fare', data=titanic)
    img_path = os.path.join("static", "pclass_vs_fare.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


"""4. Survival Rate"""
"""By class"""


def pclass_survival():
    sns.barplot(x='Pclass', y='Survived', data=titanic)
    img_path = os.path.join("static", "pclass_survival.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def pclass_survival_counts():
    return titanic.groupby(['Pclass', 'Survived'])['Survived'].count().to_dict()


"""By gender"""


def sex_survival_bar():
    sns.barplot(x='Sex', y='Survived', data=titanic)
    img_path = os.path.join("static", "sex_survival_bar.png")
    plt.savefig(img_path)
    plt.close()

    return img_path


def sex_survival_counts():
    return titanic.groupby(['Sex', 'Survived'])['Survived'].count().to_dict()


@app.route("/")
def index():
    images = {
        "Missing Value": missing_value(),
        "Age Histogram": age_histogram(),
        "Fare Boxplot": fare_boxplot(),
        "Pclass Count": pclass_count(),
        "Survival by Age": survival_by_age(),
        "Sex Survival": sex_survival(),
        "Pairplot": pairplot(),
        "Correlation Heatmap": correlation_heatmap(),
        "Pclass vs Fare Boxplot": pclass_vs_fare(),
        "Pclass vs Survival Rate": pclass_survival(),
        "Sex vs Survival Rate": sex_survival_bar(),
    }
    tables = {
        "Pclass Survival Counts": pclass_survival_counts(),
        "Sex Survival Counts": sex_survival_counts(),
    }
    return render_template("index.html", images=images, tables=tables)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
