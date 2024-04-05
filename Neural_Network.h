#pragma once

#ifndef Neural_Network
#define Neural_Network


#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <ctime>
#include <cstdlib>


using namespace std;


class NeuralNetwork
{
    private:
        string filename;

        vector<int> topology;

        vector<vector<double>> cacheLayers;
        vector<vector<double>> layers;

        vector<vector<vector<double>>> weights;
        vector<vector<double>> biases;

        double trainingRate = 0.005;


        double sigmoid(double x);
        double sigmoid_derivative(double x);

        void propogateForward(vector<double>& input);

        void propogateBackwards(const vector<double>& output, vector<vector<vector<double>>>& wG, vector<vector<double>>& bG, int num_training_examples);
        void updateWeightsAndBiases(vector<vector<vector<double>>> wGradient, vector<vector<double>> bGradient);

        vector<double> str_to_double_vec(const string& tmps);
        vector<string> str_to_str_vec(const string& tmpss);

        double random_double() const;

    public:
        struct digit
        {
            vector<double> pixels; // Input
            vector<double> answer; // Expected output
        };


        NeuralNetwork(vector<int> topology, string savetofilename);
        NeuralNetwork(string filename);

        vector<double> runNetwork(digit example);

        void train(vector<digit> trainingExamples, double trainingRate = 0.005);

        ~NeuralNetwork();
};


//
// Code Break
//

double NeuralNetwork::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

void NeuralNetwork::propogateForward(vector<double>& input)
{
    layers[0] = input;

    for (int i = 1; i < topology.size(); i++) {
        for (int j = 0; j < topology[i]; j++) {
            for (int m = 0; m < topology[i - 1]; m++) {
                layers[i][j] += layers[i - 1][m] * weights[i - 1][j][m];
            }

            layers[i][j] += biases[i][j];

            cacheLayers[i][j] = layers[i][j];
            layers[i][j] = sigmoid(layers[i][j]);
        }
    }
}

void NeuralNetwork::propogateBackwards(const vector<double>& output, vector<vector<vector<double>>>& wG, vector<vector<double>>& bG, int num_training_examples)
{
    for (int i = 0; i < topology.size() - 1; i++) {
        for (int j = 0; j < topology[i + 1]; j++) {
            for (int k = 0; k < topology[i]; k++) {
                double val = 0;

                if (i == topology.size() - 2) {
                    val += trainingRate;
                    val *= 2 * sigmoid(output[j] - sigmoid(cacheLayers[i + 1][j])) * sigmoid_derivative(cacheLayers[i + 1][j]) * sigmoid_derivative(cacheLayers[i][k]);
                } else {
                    for (int g = 0; g < topology[topology.size() - 1]; g++) {
                        double val2 = 0;
                        val2 += trainingRate * 2 * (output[g] - sigmoid(cacheLayers[topology.size() - 1][g])) * -1 * sigmoid_derivative(cacheLayers[topology.size() - 1][g]);

                        for (int h = topology.size() - 2; h > i + 1; h++) {
                            for (int m = 0; m < topology[h]; m++) {
                                val2 *= sigmoid_derivative(cacheLayers[h][m]);
                            }
                        }

                        val2 *= sigmoid_derivative(cacheLayers[i][k]);

                        val += val2;
                    }
                }

                wG[i][j][k] += val / num_training_examples;
            }
        }
    }


    for (int i = 1; i < topology.size(); i++) {
        for (int j = 0; j < topology[i]; j++) {
            double val = 0;

            if (i == topology.size() - 1) {
                val = trainingRate;
                val *= 2 * (output[j] - sigmoid(cacheLayers[i][j])) * -1 * sigmoid_derivative(cacheLayers[i][j]);
            } else {
                for (int g = 0; g < topology[topology.size() - 1]; g++) {
                    double val2 = 0;
                    val2 += trainingRate * 2 * (output[g] - sigmoid(cacheLayers[topology.size() - 1][g])) * -1 * sigmoid_derivative(cacheLayers[topology.size() - 1][g]);

                    for (int h = topology.size() - 2; h > i; h++) {
                        for (int k = 0; k < topology[h]; k++) {
                            val2 *= sigmoid_derivative(cacheLayers[h][k]);
                        }
                    }

                    val += val2;
                }
            }

            bG[i][j] += val / num_training_examples;
        }
    }
}

void NeuralNetwork::updateWeightsAndBiases(vector<vector<vector<double>>> wGradient, vector<vector<double>> bGradient)
{
    for (int i = 0; i < wGradient.size(); i++) {
        for (int j = 0; j < wGradient[i].size(); j++) {
            for (int m = 0; m < wGradient[i][j].size(); m++) {
                weights[i][j][m] -= wGradient[i][j][m];
            }
        }
    }

    for (int i = 0; i < bGradient.size(); i++) {
        for (int j = 0; j < bGradient[i].size(); j++) {
            biases[i][j] -= bGradient[i][j];
        }
    }
}

vector<double> NeuralNetwork::str_to_double_vec(const string& tmps)
{
    vector<double> v;

    string tmp;

    for (char c : tmps) {
        if (c == ' ') {
            v.push_back(stod(tmp));
            tmp.clear();
        } else {
            tmp += c;
        }
    }

    return v;
}

vector<string> NeuralNetwork::str_to_str_vec(const string& tmpss)
{
    vector<string> v;

    string tmps;

    for (char c : tmpss) {
        if (c == '/') {
            v.push_back(tmps);
            tmps.clear();
        } else {
            tmps += c;
        }
    }

    return v;
}

double NeuralNetwork::random_double() const
{
    double d = 0;

    if (rand() % 2 == 0) {
        d += rand() % 10;
    } else {
        d -= rand() % 10;
    }

    return d;
}

//
// Break
//

NeuralNetwork::NeuralNetwork(vector<int> topology, string savetofilename)
: topology(topology), filename(savetofilename)
{
    srand(time(nullptr));

    for (int i = 0; i < topology.size(); i++) {
        layers.push_back(vector<double>(topology[i]));
        cacheLayers.push_back(vector<double>(topology[i]));

        if (i != 0) {
            biases.push_back(vector<double>(topology[i]));
        }

        if (i != topology.size() - 1) {
            weights.push_back(vector<vector<double>>(topology[i], vector<double>(topology[i - 1])));
        }
    }

    for (auto& m : weights) {
        for (auto& ws : m) {
            for (auto& w : ws) {
                w = random_double();
            }
        }
    }

    for (auto& v : biases) {
        for (auto& b : v) {
            b = random_double();
        }
    }
}

NeuralNetwork::NeuralNetwork(string filename)
: filename(filename)
{
    ifstream file(filename);

    string top;
    getline(file, top);

    string tmp;

    for (char c : top) {
        if (c == ' ') {
            topology.push_back(stoi(tmp));
            tmp.clear();
        } else {
            tmp += c;
        }
    }

    for (int i = 0; i < topology.size(); i++) {
        layers.push_back(vector<double>(topology[i]));
        cacheLayers.push_back(vector<double>(topology[i]));
    }

    string wghts;
    getline(file, wghts);

    string tmpss;

    for (char c : wghts) {
        if (c == '$') {
            vector<vector<double>> v;
            vector<string> t = str_to_str_vec(tmpss);

            for (auto s : t) {
                v.push_back(str_to_double_vec(s));
            }

            weights.push_back(v);
            tmpss.clear();
        } else {
            tmpss += c;
        }
    }


    string biass;
    getline(file, biass);

    vector<string> s = str_to_str_vec(biass);

    for (auto ss : s) {
        biases.push_back(str_to_double_vec(ss));
    }

    file.close();
}

vector<double> NeuralNetwork::runNetwork(digit example)
{
    if (example.pixels.size() != topology[0]) {
        throw runtime_error("NeuralNetwork::runNetwork can only accept digit parameters compatible with the input layer");
    }

    propogateForward(example.pixels);

    return layers[topology.size() - 1];
}

void NeuralNetwork::train(vector<digit> trainingExamples, double trainingRate = 0.005)
{
    trainingRate = trainingRate;

    vector<vector<vector<double>>> wG(topology.size() - 1);
    vector<vector<double>> bG(topology.size() - 1);
    for (int i = 0; i < topology.size(); i++) {
        if (i != 0) {
            bG.push_back(vector<double>(topology[i]));
        }

        if (i != topology.size() - 1) {
            wG.push_back(vector<vector<double>>(topology[i], vector<double>(topology[i - 1])));
        }
    }

    for (digit d : trainingExamples) {
        if (d.pixels.size() != topology[0]) {
            throw runtime_error("NeuralNetwork::runNetwork can only accept digit parameters compatible with the input layer");
        }

        propogateForward(d.pixels);

        propogateBackwards(d.answer, wG, bG, trainingExamples.size());
    }

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int m = 0; m < weights[i][j].size(); m++) {
                weights[i][j][m] -= wG[i][j][m];
            }
        }
    }

    for (int i = 0; i < biases.size(); i++) {
        for (int j = 0; j < biases[i].size(); j++) {
            biases[i][j] -= bG[i][j];
        }
    }
}

NeuralNetwork::~NeuralNetwork()
{
    ofstream file(filename);

    if (file.fail()) {
        throw runtime_error("System failed to create file: " + filename);
    }

    for (int i = 0; i < topology.size(); i++) {
        file << topology[i];
        if (i != topology.size() - 1) {
            file << " ";
        }
    }
    file << "\n";


    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                file << weights[i][j][k];
                if (k != weights[i][j].size() - 1) {
                    file << " ";
                }
            }
            if (j != weights[i].size() - 1) {
                file << "/";
            }
        }
        if (i != weights.size() - 1) {
            file << "$";
        }
    }
    file << "\n";


    for (int i = 0; i < biases.size(); i++) {
        for (int j = 0; j < biases[i].size(); j++) {
            file << biases[i][j];
            if (j != biases[i].size()) {
                file << " ";
            }
        }
        if (i != biases.size()) {
            file << "/";
        }
    }


    file.close();
}


#endif // Neural_Network