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
#include <iomanip>
#include <iostream>


using namespace std;


class NeuralNetwork
{
    private:
        string filename;

        vector<int> topology;

        vector<vector<double>>* cacheLayers;
        vector<vector<double>>* layers;

        vector<vector<vector<double>>>* weights;
        vector<vector<double>>* biases;

        double trainingRate;


        double sigmoid(double x) const;
        double sigmoid_derivative(double x) const;

        void propogateForward(const vector<double>& input);

        void propogateBackwards(const vector<double>& output, int num_training_examples);

        vector<double> str_to_double_vec(const string& tmps);

        double random_double() const;

    public:
        struct digit
        {
            vector<double> pixels; // Input
            vector<double> answer; // Expected output

            digit(vector<double> i, vector<double> o)
            : pixels(i), answer(o)
            {
                
            }
        };


        NeuralNetwork(vector<int> topology, string savetofilename);
        NeuralNetwork(string filename);

        vector<double> runNetwork(digit example);

        void train(vector<digit> trainingExamples, double trainingR);
        void train(vector<digit> trainingExamples);

        ~NeuralNetwork();
};


//
// Code Break
//

double NeuralNetwork::sigmoid(double x) const
{
	return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x) const
{
	return sigmoid(x) * (1 - sigmoid(x));
}

void NeuralNetwork::propogateForward(const vector<double>& input)
{
    for (int i = 0; i < topology[i]; i++) {
        layers->at(0)[i] = input[i];
        cacheLayers->at(0)[i] = input[i];
    }

    for (int i = 1; i < topology.size(); i++) {
        for (int j = 0; j < topology[i]; j++) {
            layers->at(i)[j] = 0;

            for (int m = 0; m < topology[i - 1]; m++) {
                layers->at(i)[j] += layers->at(i - 1)[m] * weights->at(i - 1)[j][m];
            }

            layers->at(i)[j] += biases->at(i - 1)[j];

            cacheLayers->at(i)[j] = layers->at(i)[j];
            layers->at(i)[j] = sigmoid(layers->at(i)[j]);
        }
    }
}

void NeuralNetwork::propogateBackwards(const vector<double>& output, int num_training_examples)
{
    for (int i = 0; i < topology.size() - 1; i++) {
        for (int j = 0; j < topology[i + 1]; j++) {
            for (int k = 0; k < topology[i]; k++) {
                double val = 0;

                if (i == topology.size() - 2) {
                    val += trainingRate;
                    val *= 2 * sigmoid(output[j] - sigmoid(cacheLayers->at(i + 1)[j])) * sigmoid_derivative(cacheLayers->at(i + 1)[j]) * sigmoid_derivative(cacheLayers->at(i)[k]);
                } else {
                    for (int g = 0; g < topology[topology.size() - 1]; g++) {
                        double val2 = 0;
                        val2 += trainingRate * 2 * (output[g] - sigmoid(cacheLayers->at(topology.size() - 1)[g])) * -1 * sigmoid_derivative(cacheLayers->at(topology.size() - 1)[g]);

                        for (int h = topology.size() - 2; h > i + 1; h--) {
                            for (int m = 0; m < topology[h]; m++) {
                                val2 *= sigmoid_derivative(cacheLayers->at(h)[m]);
                            }
                        }

                        val2 *= sigmoid_derivative(cacheLayers->at(i)[k]);

                        val += val2;
                    }
                }

                weights->at(i)[j][k] -= val / num_training_examples;
            }
        }
    }


    for (int i = 0; i < topology.size() - 1; i++) {
        for (int j = 0; j < topology[i]; j++) {
            double val = 0;

            if (i == topology.size() - 1) {
                val = trainingRate;
                val *= 2 * (output[j] - sigmoid(cacheLayers->at(i)[j])) * -1 * sigmoid_derivative(cacheLayers->at(i)[j]);
            } else {
                for (int g = 0; g < topology[topology.size() - 1]; g++) {
                    double val2 = 0;
                    val2 += trainingRate * 2 * (output[g] - sigmoid(cacheLayers->at(topology.size() - 1)[g])) * -1 * sigmoid_derivative(cacheLayers->at(topology.size() - 1)[g]);

                    for (int h = topology.size() - 2; h > i; h--) {
                        for (int k = 0; k < topology[h]; k++) {
                            val2 *= sigmoid_derivative(cacheLayers->at(h)[k]);
                        }
                    }

                    val += val2;
                }
            }

            biases->at(i)[j] -= val / num_training_examples;
        }
    }
}

vector<double> NeuralNetwork::str_to_double_vec(const string& tmps)
{
    vector<double> v;

    string tmp;

    for (int i = 0; i < tmps.size(); i++) {
        if (tmps[i] == ' ') {
            v.push_back(stod(tmp));
            tmp.clear();
        } else if (i == tmps.size() - 1) {
            tmp += tmps[i];
            v.push_back(stod(tmp));
        } else {
            tmp += tmps[i];
        }
    }

    return v;
}

double NeuralNetwork::random_double() const
{
    double d = 0;

    if (rand() % 2 == 0) {
        d += static_cast<double>(rand() % 100);
    } else {
        d -= static_cast<double>(rand() % 100);
    }

    double d2 = static_cast<double>(1) / (static_cast<double>(rand() % 5) + static_cast<double>(10));

    d *= d2;

    return d;
}

//
// Break
//

NeuralNetwork::NeuralNetwork(vector<int> topology, string savetofilename)
: filename(savetofilename), topology(topology), cacheLayers(new vector<vector<double>>), layers(new vector<vector<double>>), weights(new vector<vector<vector<double>>>), biases(new vector<vector<double>>)
{
    srand(time(nullptr));

    for (int i = 0; i < topology.size(); i++) {
        vector<double> lTemp;
        vector<double> cacheTemp;
        for (int j = 0; j < topology[i]; j++) {
            double d = 0;
            lTemp.push_back(d);
            cacheTemp.push_back(d);
        }
        layers->push_back(lTemp);
        cacheLayers->push_back(cacheTemp);

        if (i > 0) {
            vector<vector<double>> wTemp;
            for (int j = 0; j < topology[i]; j++) {
                vector<double> wTemp2;
                for (int l = 0; l < topology[i - 1]; l++) {
                    wTemp2.push_back(random_double());
                }

                wTemp.push_back(wTemp2);
            }

            vector<double> bTemp;
            for (int j = 0; j < topology[i]; j++) {
                bTemp.push_back(random_double());
            }

            weights->push_back(wTemp);
            biases->push_back(bTemp);
        }
    }
}

NeuralNetwork::NeuralNetwork(string filename)
: filename(filename), cacheLayers(new vector<vector<double>>), layers(new vector<vector<double>>), weights(new vector<vector<vector<double>>>), biases(new vector<vector<double>>)
{
    ifstream file(filename);

    if (file.fail()) {
        throw runtime_error("File " + filename + " failed to load properly");
    }

    string top;
    getline(file, top);

    string tmp;

    for (int i = 0; i < top.size(); i++) {
        if (top[i] == ' ') {
            topology.push_back(stoi(tmp));
            tmp.clear();
        } else if (i == top.size() - 1) {
            tmp += top[i];
            topology.push_back(stoi(tmp));
        } else {
            tmp += top[i];
        }
    }

    for (int i = 0; i < topology.size(); i++) {
        layers->push_back(vector<double>(topology[i]));
        cacheLayers->push_back(vector<double>(topology[i]));
    }

    string wghts;
    getline(file, wghts);

    vector<string> v;
    string temp;
    int layerIdx = 1;
    int countSpaces = 0;

    for (int i = 0; i < wghts.size(); i++) {
        if (wghts[i] == ' ') {
            countSpaces++;
            if (countSpaces == (topology[layerIdx] * topology[layerIdx - 1])) {
                countSpaces = 0;
                layerIdx++;
                v.push_back(temp);
                temp.clear();
                continue;
            }
            temp += wghts[i];
        } else if (i == wghts.size() - 1) {
            temp += wghts[i];
            v.push_back(temp);
        } else {
            temp += wghts[i];
        }
    }

    int layerIdx2 = 0;

    for (auto& s : v) {
        vector<vector<double>> v2;

        string temp2;
        int countSpaces2 = 0;

        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ' ') {
                countSpaces2++;
                if (countSpaces2 == topology[layerIdx2]) {
                    countSpaces2 = 0;
                    v2.push_back(str_to_double_vec(temp2));
                    temp2.clear();
                    continue;
                }
                temp2 += s[i];
            } else if (i == s.size() - 1) {
                temp2 += s[i];
                v2.push_back(str_to_double_vec(temp2));
            } else {
                temp2 += s[i];
            }
        }

        weights->push_back(v2);
        layerIdx2++;
    }


    string biass;
    getline(file, biass);

    string temp3;
    int layerIdx3 = 1;
    int countSpaces3 = 0;

    for (int i = 0; i < biass.size(); i++) {
        if (biass[i] == ' ') {
            countSpaces3++;
            if (countSpaces3 == topology[layerIdx3]) {
                countSpaces3 = 0;
                layerIdx3++;
                biases->push_back(str_to_double_vec(temp3));
                temp3.clear();
                continue;
            }
            temp3 += biass[i];
        } else if (i == biass.size() - 1) {
            temp3 += biass[i];
            biases->push_back(str_to_double_vec(temp3));
        } else {
            temp3 += biass[i];
        }
    }

    file.close();
}

vector<double> NeuralNetwork::runNetwork(digit example)
{
    if ((example.pixels).size() != topology[0]) {
        throw runtime_error("NeuralNetwork::runNetwork can only accept digit parameters compatible with the input layer");
    }

    propogateForward(example.pixels);

    return layers->at(topology.size() - 1);
}

void NeuralNetwork::train(vector<digit> trainingExamples, double trainingR)
{
    trainingRate = trainingR;

    for (auto& d : trainingExamples) {
        if (d.pixels.size() != topology[0] || d.answer.size() != topology[topology.size() - 1]) {
            throw runtime_error("NeuralNetwork::train can only accept digit parameters compatible with the input & output layers");
        }

        propogateForward(d.pixels);


        propogateBackwards(d.answer, trainingExamples.size());
    }
}

void NeuralNetwork::train(vector<digit> trainingExamples)
{
    train(trainingExamples, 0.005);
}

NeuralNetwork::~NeuralNetwork()
{
    ofstream file("/home/owen_n/CMPT_135/1PersonalProjects/Capstone Project/" + filename);

    for (int i = 0; i < topology.size(); i++) {
        file << topology[i];
        if (i != topology.size() - 1) {
            file << " ";
        }
    }
    file << "\n";


    for (int i = 0; i < weights->size(); i++) {
        for (int j = 0; j < weights->at(i).size(); j++) {
            for (int k = 0; k < weights->at(i)[j].size(); k++) {
                file << fixed << setprecision(10) << weights->at(i)[j][k];
                if (!(k == weights->at(i)[j].size() - 1 && j == weights->at(i).size() - 1 && i == weights->size() - 1)) {
                    file << " ";
                }
            }
        }
    }
    file << "\n";


    for (int i = 0; i < biases->size(); i++) {
        for (int j = 0; j < biases->at(i).size(); j++) {
            file << fixed << setprecision(10) << biases->at(i)[j];
            if (!(j == biases->at(i).size() - 1 && i == biases->size() - 1)) {
                file << " ";
            }
        }
        
    }


    file.close();

    delete[] layers;
    delete[] cacheLayers;
    delete[] weights;
    delete[] biases;
}


#endif // Neural_Network