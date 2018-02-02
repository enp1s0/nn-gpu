#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include "matrix_array.h"

// 訓練データの読み込みクラス
//

class DataLoader{
	int data_count;
	std::mt19937 mt;
	mtk::MatrixXf sata,labels;
public:
	int load(std::string path);
	DataLoader();
	~DataLoader();
};
