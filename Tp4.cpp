#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <cstdlib>
#include <limits>
#include <string>

using namespace std;

// --------------------------------------------------------
// Question 1: Implémentez la classe mère TimeSeriesGenerator
class TimeSeriesGenerator {
public:
    int seed;

    TimeSeriesGenerator() : seed(0) {}
    TimeSeriesGenerator(int s) : seed(s) {}

    virtual vector<double> generateTimeSeries(int size) = 0;

    static void printTimeSeries(const vector<double>& series) {
        for (double val : series) {
            cout << val << " ";
        }
        cout << endl;
    }
};

// --------------------------------------------------------
// Question 2: Implémentez la classe GaussianGenerator qui hérite de TimeSeriesGenerator
class GaussianGenerator : public TimeSeriesGenerator {
public:
    double mean, stddev;

    GaussianGenerator(double m = 0.0, double s = 1.0) : mean(m), stddev(s) {}

    vector<double> generateTimeSeries(int size) override {
        vector<double> series(size);
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(mean, stddev);

        for (int i = 0; i < size; ++i) {
            series[i] = dist(gen);
        }

        return series;
    }
};

// --------------------------------------------------------
// Question 3: Implémentez la classe StepGenerator qui hérite de TimeSeriesGenerator
class StepGenerator : public TimeSeriesGenerator {
public:
    StepGenerator() {}

    vector<double> generateTimeSeries(int size) override {
        vector<double> series(size);
        series[0] = 0;

        for (int i = 1; i < size; ++i) {
            if (rand() % 2 == 0) {
                series[i] = rand() % 101;  // Saut aléatoire entre 0 et 100
            } else {
                series[i] = series[i - 1];  // Maintenir la valeur précédente
            }
        }

        return series;
    }
};

// --------------------------------------------------------
// Question 4: Implémentez la classe SinWaveGenerator qui hérite de TimeSeriesGenerator
class SinWaveGenerator : public TimeSeriesGenerator {
public:
    double amplitude, frequency, phase;

    SinWaveGenerator(double A = 1.0, double omega = 1.0, double phi = 0.0) 
        : amplitude(A), frequency(omega), phase(phi) {}

    vector<double> generateTimeSeries(int size) override {
        vector<double> series(size);
        double step = (2 * M_PI) / size;

        for (int i = 0; i < size; ++i) {
            series[i] = amplitude * sin(frequency * i * step + phase);
        }

        return series;
    }
};

// --------------------------------------------------------
// Question 5: Implémentez la classe TimeSeriesDataset
class TimeSeriesDataset {
public:
    bool znormalize;
    bool isTrain;
    vector<vector<double>> data;
    vector<int> labels;
    int maxLength;
    int numberOfSamples;

    TimeSeriesDataset(bool zn = false, bool train = true)
        : znormalize(zn), isTrain(train), maxLength(0), numberOfSamples(0) {}

    void addTimeSeries(const vector<double>& series, int label) {
        if (znormalize) {
            vector<double> normalized = zNormalize(series);
            data.push_back(normalized);
        } else {
            data.push_back(series);
        }
        labels.push_back(label);
        maxLength = max(maxLength, (int)series.size());
        numberOfSamples++;
    }

private:
    // Question 6: Fonction de normalisation Z
    vector<double> zNormalize(const vector<double>& series) {
        vector<double> normalized(series.size());
        double mean = accumulate(series.begin(), series.end(), 0.0) / series.size();
        double stddev = 0.0;

        for (double val : series) {
            stddev += pow(val - mean, 2);
        }
        stddev = sqrt(stddev / series.size());

        for (int i = 0; i < series.size(); ++i) {
            normalized[i] = (series[i] - mean) / stddev;
        }

        return normalized;
    }
};

// --------------------------------------------------------
// Question 7: Fonction de calcul de la distance Euclidienne
double euclideanDistance(const vector<double>& x, const vector<double>& y) {
    double sum = 0.0;
    int minSize = min(x.size(), y.size());

    for (int i = 0; i < minSize; ++i) {
        sum += pow(x[i] - y[i], 2);
    }

    return sqrt(sum);
}

// --------------------------------------------------------
// Question 9: Implémentation de la classe KNN (K-Nearest Neighbors)
class KNN {
private:
    int k;
    string similarityMeasure;

public:
    KNN(int k_val, string simMeasure)
        : k(k_val), similarityMeasure(simMeasure) {}

    double evaluate(const TimeSeriesDataset& trainData, const TimeSeriesDataset& testData, const vector<int>& groundTruth) {
        int correct = 0;
        for (int i = 0; i < testData.numberOfSamples; ++i) {
            vector<pair<double, int>> distances;

            for (int j = 0; j < trainData.numberOfSamples; ++j) {
                double dist = (similarityMeasure == "euclidean_distance") 
                                ? euclideanDistance(testData.data[i], trainData.data[j]) 
                                : dynamicTimeWarping(testData.data[i], trainData.data[j]);

                distances.push_back({dist, trainData.labels[j]});
            }

            sort(distances.begin(), distances.end());

            map<int, int> classCount;
            for (int i = 0; i < k; ++i) {
                classCount[distances[i].second]++;
            }

            int predictedClass = max_element(classCount.begin(), classCount.end(),
                                              [](const pair<int, int>& a, const pair<int, int>& b) { return a.second < b.second; })->first;

            if (predictedClass == groundTruth[i]) {
                correct++;
            }
        }

        return static_cast<double>(correct) / testData.numberOfSamples;
    }
};

// --------------------------------------------------------
// Question 10: Fonction main pour exécuter le programme
int main() {
    TimeSeriesDataset trainData(true, true), testData(true, false);
    GaussianGenerator gsg;
    SinWaveGenerator swg;
    StepGenerator stg;

    // Ajout des séries temporelles d'entraînement
    vector<double> gaussian1 = gsg.generateTimeSeries(11);
    trainData.addTimeSeries(gaussian1, 0);
    vector<double> gaussian2 = gsg.generateTimeSeries(11);
    trainData.addTimeSeries(gaussian2, 0);

    vector<double> sin1 = swg.generateTimeSeries(11);
    trainData.addTimeSeries(sin1, 1);
    vector<double> sin2 = swg.generateTimeSeries(11);
    trainData.addTimeSeries(sin2, 1);

    vector<double> step1 = stg.generateTimeSeries(11);
    trainData.addTimeSeries(step1, 2);
    vector<double> step2 = stg.generateTimeSeries(11);
    trainData.addTimeSeries(step2, 2);

    // Ajout des séries temporelles de test et de la vérité terrain
    vector<int> groundTruth;
    testData.addTimeSeries(gsg.generateTimeSeries(11)); groundTruth.push_back(0);
    testData.addTimeSeries(swg.generateTimeSeries(11)); groundTruth.push_back(1);
    testData.addTimeSeries(stg.generateTimeSeries(11)); groundTruth.push_back(2);

    // Évaluation KNN avec différents paramètres k et mesures de distance
    KNN knn1(1, "dtw");
    cout << "Précision avec DTW (k=1): " << knn1.evaluate(trainData, testData, groundTruth) << endl;

    KNN knn2(2, "euclidean_distance");
    cout << "Précision avec Euclidean (k=2): " << knn2.evaluate(trainData, testData, groundTruth) << endl;

    KNN knn3(3, "euclidean_distance");
    cout << "Précision avec Euclidean (k=3): " << knn3.evaluate(trainData, testData, groundTruth) << endl;

    return 0;
}
