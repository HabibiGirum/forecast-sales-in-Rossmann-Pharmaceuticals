
# forecast-sales-in-Rossmann-Pharmaceuticals

![Build Status](https://github.com/HabibiGirum/forecast-sales-in-Rossmann-Pharmaceuticals/actions/workflows/unittest.yml/badge.svg)

## Overview
You work at Rossmann Pharmaceuticals as a Machine Learning Engineer. The finance team wants to forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgment to forecast sales.

The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.

our job is  to build and serve an end-to-end product that delivers this prediction to analysts in the finance team. 

## Dataset

Files
**train.csv** - historical data including Sales
**test.csv** - historical data excluding Sales
**sample_submission.csv** - a sample submission file in the correct format
**store.csv** - supplemental information about the stores

Data fields
Most of the fields are self-explanatory. The following are descriptions for those that aren't.

**Id** - an Id that represents a (Store, Date) duple within the test set

**Store** - a unique Id for each store

**Sales** - the turnover for any given day (this is what you are predicting)

**Customers** - the number of customers on a given day

**Open** - an indicator for whether the store was open: 0 = closed, 1 = open

**StateHoliday** - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

**SchoolHoliday** - indicates if the (Store, Date) was affected by the closure of public schools

**StoreType** - differentiates between 4 different store models: a, b, c, d

**Assortment** - describes an assortment level: a = basic, b = extra, c = extended

**CompetitionDistance** - distance in meters to the nearest competitor store

**CompetitionOpenSince[Month/Year]** - gives the approximate year and month of the time the nearest competitor was opened

**Promo** - indicates whether a store is running a promo on that day

**Promo2** - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
Promo2Since[Year/Week]** - describes the year and calendar week when the store started participating in Promo2

**PromoInterval** - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store


## Installation

To run the code in this repository, follow these steps:

### Prerequisites

Make sure you have Python 3.11. You can check your Python version by running:

```bash
python --version
```

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/HabibiGirum/forecast-sales-in-Rossmann-Pharmaceuticals.git
cd forecast-sales-in-Rossmann-Pharmaceuticals
```

### Create a Virtual Environment (Optional but Recommended)

Create a virtual environment to isolate the project dependencies:

```bash
python3 -m venv env
```

Activate the virtual environment:

- On Windows:
  ```bash
  .\env\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source env/bin/activate
  ```

### Install Required Packages

Install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

### Running the Analysis

Once you have set up the environment and installed the dependencies, you can run the EDA scripts:

```notbooks/analysis.ipynb```

## About the Analysis

This repository includes EDA:



## Further documentation :
[click me](https://drive.google.com/file/d/15aGTZZdOCfE5vhIW5yV4cRS2wzHES72a/view?usp=sharing)


## Author  
GitHub: [HabibiGirum](https://github.com/HabibiGirum)

Email:  habtamugirum478@gmail.com