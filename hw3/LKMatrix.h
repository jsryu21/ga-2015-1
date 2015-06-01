#ifndef LK_MATRIX
#define LK_MATRIX

#include <vector>

using namespace std;

class LKMatrix {
  public:
    int size;
    LKMatrix(vector<pair<double, double> > &coords, vector<int> &ids);
    LKMatrix(const std::vector< std::vector< double > >& edgeDistances);
    vector<int> getCurrentTour();
    double getCurrentTourDistance();
    void optimizeTour();
    void OptimizeTour(const std::vector< int >& tour, std::vector< int >& tour_opt);
    void printTour();
    void printTourIds();

  private:
    vector<int> tour;
    vector<vector<int> > edgeFlags;
    vector<pair<double, double> > coords;
    vector<int> ids;
    void joinLocations(int i, int j);
    vector<vector<double> > edgeDistances;
    void LKMove(int tourStart);
    void reverse(int start, int end);
    bool isTour();
};

#endif
