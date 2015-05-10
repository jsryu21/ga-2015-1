#include <vector>

class TestCase {
    /*****************************************************************
      Input variables
     *****************************************************************/
    public:
        TestCase();
        void PrintTestCase();
        // The number of locations
        int NumLocations;
        // Dist[i][j] := the Distance between (x[i], y[i]) and (x[j], y[j])
        // will be automatically calculated
        std::vector< std::vector< double > > Dist;
        // Time limit for the test case
        long long TimeLimit;
    private:
        void Init();
        // (x[i], y[i]) := the i-th location
        std::vector< double > x;
        std::vector< double > y;
};
