#pragma once

#include "EasyBMP.h"

#include <string>
#include <vector>

typedef std::vector<std::pair<BMP *, int>> TDataSet;
typedef std::vector<std::pair<std::string, int>> TFileList;
typedef std::vector<std::pair<std::vector<float>, int>> TFeatures;
typedef std::vector<int> TLabels;
