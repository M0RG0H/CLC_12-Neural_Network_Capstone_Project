#pragma once

#ifndef Neural_Network
#define Neural_Network


#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>


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
    //...


    for (int i = topology.size() - 1; i > 0; i++) {
        for (int j = topology[i] - 1; j > 0; j++) {
            double val = 0;

            if (i == topology.size() - 1) {
                val = trainingRate;
                val *= 2 * (output[j] - sigmoid(cacheLayers[i][j])) * sigmoid_derivative(cacheLayers[i][j]);
            } else {
                for (int g = 0; g < topology[topology.size() - 1]; g++) {
                    double val2 = 0;
                    val2 += trainingRate * 2 * (output[g] - sigmoid(cacheLayers[topology.size() - 1][g]));

                    for (int h = topology.size() - 2; h >= i; h++) {
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

//
// Break
//

NeuralNetwork::NeuralNetwork(vector<int> topology, string savetofilename)
: topology(topology), filename(savetofilename)
{
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

    //...
}

NeuralNetwork::NeuralNetwork(string filename)
: filename(filename)
{
    //... Need to create topology vector

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

    //...
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

    vector<vector<vector<double>>> wG(topology.size() - 2);
    for () {
        //...
    }

    vector<vector<double>> bG(topology.size() - 2);
    for () {
        //...
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
    //...
}


#endif // Neural_Network