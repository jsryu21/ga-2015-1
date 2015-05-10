#include "test_case.h"
#include <iostream>
#include <cmath>

TestCase::TestCase() {
    Init();
}

void TestCase::Init() {
    // read the test case from stdin
    // and initialize some values such as record.f and Dist
    std::cin >> NumLocations;
    x.resize(NumLocations);
    y.resize(NumLocations);
    Dist.resize(NumLocations);
    for (int i = 0; i < NumLocations; ++i) {
        Dist[i].resize(NumLocations);
    }
    for (int i = 0; i < NumLocations; ++i) {
        std::cin >> x[i] >> y[i];
    }
    for (int i = 0; i < NumLocations; ++i) {
        for (int j = 0; j < NumLocations; ++j) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            Dist[i][j] = std::sqrt(dx * dx + dy * dy);
        }
    }
    std::cin >> TimeLimit;
}

void TestCase::PrintTestCase() {
    std::cout << "Num locations : " << NumLocations << std::endl;
    std::cout << "Locations : " << std::endl;
    for (int i = 0; i < NumLocations; ++i) {
        std::cout << "(" << x[i] << ", " << y[i] << ")" << std::endl; 
    }
    std::cout << "Time limit : " << TimeLimit << std::endl;
}

