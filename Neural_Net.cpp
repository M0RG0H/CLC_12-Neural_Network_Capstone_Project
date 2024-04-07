#include "Neural_Network.h"
#include <iostream>
#include <vector>
#include <string>

#include <random>
#include <chrono>
#include <algorithm>

#include <cassert>
#include <fstream>

using namespace std;

//Type Definitions
typedef unsigned char uchar;
typedef NeuralNetwork::digit digi;

//Function declarations
vector<vector<double>> ReadMNISTImages(const string& mnistfilepath_imgs);
vector<vector<double>> ReadMNISTLabels(const string& mnistfilepath_lbls);


int main()
{
    // vector<int> topolog{784, 16, 16, 10};
    // string filen = "NETWORK_DATA.txt";
    // NeuralNetwork network(topolog, filen);

    string filen = "NETWORK_DATA.txt";
    NeuralNetwork network(filen);

    vector<vector<double>> imgs = ReadMNISTImages("/home/owen_n/CMPT_135/1PersonalProjects/Capstone Project/Dataset/train-images-idx3-ubyte/train-images.idx3-ubyte");
    vector<vector<double>> lbls = ReadMNISTLabels("/home/owen_n/CMPT_135/1PersonalProjects/Capstone Project/Dataset/train-labels.idx1-ubyte");

    while (true) {

        int numExamples = 100;

        // digi d(imgs[1], lbls[1]);
        // vector<double> va = network.runNetwork(d);

        // for (int i = 1; i <= imgs[0].size(); i++) {
        //     cout << round(imgs[0][i]) << " ";
        //     if (i % 28 == 0) {
        //         cout << endl;
        //     }
        // }
        // cout << endl;

        // for (int i = 0; i < va.size(); i++) {
        //     cout << i << ": " << va[i] << "   ";
        // }

        // cout << endl;

        vector<digi> vec;
        for (int i = 0; i <= numExamples; i++) {
            if (imgs.at(i).size() != 784 || lbls.at(i).size() != 10) {
                cout << "Bad input avoided. Whew" << endl;
                continue;
            }

            digi d(imgs.at(i), lbls.at(i));
            vec.push_back(d);
        }
        
 
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle (vec.begin(), vec.end(), std::default_random_engine(seed));


        vector<vector<digi>> bVec;

        int count = 0;

        vector<digi> dV;
        for (auto& d : vec) {
            if (count >= 10) {
                bVec.push_back(dV);
                dV.clear();
                count = 0;
            } else {
                dV.push_back(d);
                count++;
            }
        }

        int count2 = 0;
        for (auto& v : bVec) {
            network.train(v, 10);
            count2 += 10;
            cout << "~" << count2 << " training examples gone through\n";
        }
    }

    return 0;
}

vector<vector<double>> ReadMNISTImages(const string& mnistfilepath_imgs)
{
    ifstream file(mnistfilepath_imgs, ios::binary);

    if (file.fail()) {
        throw runtime_error("Dataset from file " + mnistfilepath_imgs + " failed to load");
    }

    char magicNumber[4];
    char numOfImages[4];
    char numOfRows[4];
    char numOfCols[4];

    file.read(magicNumber, 4);
    file.read(numOfImages, 4);
    file.read(numOfRows, 4);
    file.read(numOfCols, 4);

    int numImages = (static_cast<uchar>(numOfImages[0]) << 24) | (static_cast<uchar>(numOfImages[1]) << 16) | (static_cast<uchar>(numOfImages[2]) << 8) | (static_cast<uchar>(numOfImages[3]));
    int numRows = (static_cast<uchar>(numOfRows[0]) << 24) | (static_cast<uchar>(numOfImages[1]) << 16) | (static_cast<uchar>(numOfRows[2]) << 8) | (static_cast<uchar>(numOfRows[3]));
    int numCols = (static_cast<uchar>(numOfCols[0]) << 24) | (static_cast<uchar>(numOfCols[1]) << 16) | (static_cast<uchar>(numOfCols[2]) << 8) | (static_cast<uchar>(numOfCols[3]));

    vector<vector<double>> result;

    for (int i = 0; i < numImages; i++) {
        vector<uchar> image(numRows * numCols);

        file.read((char*)(image.data()), numRows * numCols);

        vector<double> image_proper(numRows * numCols);

        for (int i = 0; i < image.size(); i++) {
            image_proper[i] = (double)image[i] / static_cast<double>(256);
        }

        result.push_back(image_proper);
    }

    file.close();

    return result;
}

vector<vector<double>> ReadMNISTLabels(const string& mnistfilepath_lbls)
{
    ifstream file(mnistfilepath_lbls, ios::binary);

    if (file.fail()) {
        throw runtime_error("Dataset from file " + mnistfilepath_lbls + " failed to load");
    }

    char magicNumber[4];
    char numOfImages[4];

    file.read(magicNumber, 4);
    file.read(numOfImages, 4);

    int numImages = (static_cast<uchar>(numOfImages[0]) << 24) | (static_cast<uchar>(numOfImages[1]) << 16) | (static_cast<uchar>(numOfImages[2]) << 8) | (static_cast<uchar>(numOfImages[3]));

    vector<vector<double>> result;

    for (int i = 0; i < numImages; i++) {
        vector<uchar> lbl(1);

        file.read((char*)(lbl.data()), 1);

        vector<double> lbl_proper(10);

        for (int i = 0; i < 10; i++) {
            if (i == (int)lbl[0]) {
                lbl_proper[i] = 1;
            } else {
                lbl_proper[i] = 0;
            }
        }

        result.push_back(lbl_proper);
    }

    file.close();

    return result;
}