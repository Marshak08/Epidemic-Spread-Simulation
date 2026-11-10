#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
using namespace std;

const int population = 1000;
const int days = 30;
const int threads = 4;
const int contacts = 10;
const int recoveryDays = 7;

const double infectionRate = 0.08;

const int healthy = 0;
const int infected = 1;
const int recovered = 2;

struct Statistics {
    int healthy = 0;
    int infected = 0;
    int recovered = 0;
};

void makeContacts(vector<vector<int>>& contactsList,
    vector<vector<double>>& chances) {

    mt19937 gen(42);

    uniform_int_distribution<int> person(0, population - 1);
    uniform_real_distribution<double> chance(0.0, 1.0);

    for (int day = 0; day < days; day++) {
        for (int i = 0; i < population; i++) {
            for (int j = 0; j < contacts; j++) {
                int index = i * contacts + j;
                contactsList[day][index] = person(gen);
                chances[day][index] = chance(gen);
            }
        }
    }
}

vector<Statistics> simulation(
    bool parallel,
    const vector<vector<int>>& contactsList,
    const vector<vector<double>>& chances) {

    vector<int> state(population, healthy);
    vector<int> nextState(population, healthy);

    vector<int> infectionDays(population, 0);
    vector<int> nextDays(population, 0);

    state[0] = infected;

    vector<Statistics> results;

    Statistics first;
    first.healthy = population - 1;
    first.infected = 1;

    results.push_back(first);

    for (int day = 0; day < days; day++) {
        #pragma omp parallel for num_threads(threads) if(parallel)

        for (int i = 0; i < population; i++) {
            if (state[i] == infected) {
                nextDays[i] = infectionDays[i] + 1;

                if (nextDays[i] >= recoveryDays) {
                    nextState[i] = recovered;
                    nextDays[i] = 0;
                }
                else {
                    nextState[i] = infected;
                }
            }

            else if (state[i] == recovered) {
                nextState[i] = recovered;
                nextDays[i] = 0;
            }

            else {
                bool newInfection = false;

                for (int j = 0; j < contacts; j++) {
                    int index = i * contacts + j;
                    int person = contactsList[day][index];

                    if (state[person] == infected &&
                        chances[day][index] < infectionRate) {
                        newInfection = true;
                        break;
                    }
                }

                if (newInfection) {
                    nextState[i] = infected;
                    nextDays[i] = 0;
                }
                else {
                    nextState[i] = healthy;
                    nextDays[i] = 0;
                }
            }
        }

        state.swap(nextState);
        infectionDays.swap(nextDays);

        Statistics result;

        for (int i = 0; i < population; i++) {
            if (state[i] == healthy)
                result.healthy++;

            else if (state[i] == infected)
                result.infected++;

            else
                result.recovered++;
        }

        results.push_back(result);
    }

    return results;
}

int main() {
    omp_set_dynamic(0);

    vector<vector<int>> contactsList(
        days, vector<int>(population * contacts)
    );

    vector<vector<double>> chances(
        days, vector<double>(population * contacts)
    );

    makeContacts(contactsList, chances);

    cout << "Epidemic simulation" << endl;
    cout << "Population: " << population << endl;
    cout << "Days: " << days << endl;
    cout << "Threads: " << threads << endl;

    auto start = chrono::steady_clock::now();

    vector<Statistics> sequential =
        simulation(false, contactsList, chances);

    auto finish = chrono::steady_clock::now();

    double sequentialTime =
        chrono::duration<double, milli>(finish - start).count();

    start = chrono::steady_clock::now();

    vector<Statistics> parallel =
        simulation(true, contactsList, chances);

    finish = chrono::steady_clock::now();

    double parallelTime =
        chrono::duration<double, milli>(finish - start).count();

    cout << "\nDaily results" << endl;
    cout << "Day  Healthy  Infected  Recovered" << endl;

    for (int day = 0; day <= days; day++) {

        cout << day << "    "
            << parallel[day].healthy << "       "
            << parallel[day].infected << "        "
            << parallel[day].recovered << endl;
    }

    cout << "\nChecking results..." << endl;

    bool correct = true;

    for (int i = 0; i <= days; i++) {

        if (sequential[i].healthy != parallel[i].healthy ||
            sequential[i].infected != parallel[i].infected ||
            sequential[i].recovered != parallel[i].recovered) {

            correct = false;
        }
    }

    if (correct)
        cout << "Both versions give the same results." << endl;
    else
        cout << "Results are different." << endl;

    double speedup = sequentialTime / parallelTime;
    double efficiency = speedup / threads * 100;

    cout << "\nPerformance results" << endl;

    cout << "Sequential time: "
        << sequentialTime << " ms" << endl;

    cout << "Parallel time: "
        << parallelTime << " ms" << endl;

    cout << "Speedup: " << speedup << endl;
    cout << "Efficiency: " << efficiency << "%" << endl;

    return 0;
}
