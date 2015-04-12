#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include "test_case.h"

// https://isocpp.org/wiki/faq/pointers-to-members
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))

const static int PSIZE = 100; // Size of the population
const static int ROULETTE_SELECTION_PRESSURE_K = 3; // 3 ~ 4
const static double TOURNAMENT_SELECTION_PRESSURE_T = 0.5;
// http://www.complex-systems.com/pdf/09-3-2.pdf
// http://en.wikipedia.org/wiki/Tournament_selection
const static int GENERAL_TOURNAMENT_SELECTION_PRESSURE_K = 5;

struct Solution
{
    Solution(int len);
    std::vector< int > Chromosome;
    double Fitness;
    bool operator <(const Solution& solution) const {
        return Fitness < solution.Fitness;
    }
};

Solution::Solution(int len) : Chromosome(std::vector< int >(len)),
    Fitness(std::numeric_limits< double >::max())
{
}

class SteadyStateGA {
    /*****************************************************************
      GA variables and functions
      Note that the representation is currently order-based.
      A chromosome will be a permutation of (0, 1, ..., N-1).
     *****************************************************************/
    public:
        typedef void (SteadyStateGA::*EvaluateFn)(Solution& s);
        typedef void (SteadyStateGA::*GenerateRandomSolutionFn)(Solution& s);
        typedef void (SteadyStateGA::*PreprocessFn)();
        typedef void (SteadyStateGA::*SelectionFn)(Solution& s);
        typedef void (SteadyStateGA::*CrossoverFn)(const Solution& p1, const Solution& p2, Solution& c);
        typedef void (SteadyStateGA::*MutationFn)(Solution& s);
        typedef void (SteadyStateGA::*ReplacementFn)(const Solution& s);
        SteadyStateGA(const TestCase& testCase
                , EvaluateFn Evaluate_
                , GenerateRandomSolutionFn GenerateRandomSolution_
                , PreprocessFn Preprocess_
                , SelectionFn Selection_
                , CrossoverFn Crossover_
                , MutationFn Mutation_
                , ReplacementFn Replacement_);
        void Evaluate1(Solution& s);
        void GenerateRandomSolution1(Solution& s);
        void Preprocess1();
        void Selection1(Solution& s);
        void Selection2(Solution& s);
        void Selection3(Solution& s);
        void Selection4(Solution& s);
        void Crossover1(const Solution& p1, const Solution& p2, Solution& c);
        void Mutation1(Solution& s);
        void Replacement1(const Solution& offspr);
        void GA();
        void Answer();
        void PrintAllSolutions();
    private:
        EvaluateFn Evaluate;
        GenerateRandomSolutionFn GenerateRandomSolution;
        PreprocessFn Preprocess;
        SelectionFn Selection;
        CrossoverFn Crossover;
        MutationFn Mutation;
        ReplacementFn Replacement;
        void PrintSolution(const Solution& s);
        void Normalize(Solution& s);

        int solutionLen;
        const std::vector< std::vector< double > >& solutionDist;
        long long timeLimit;
        // population of solutions
        std::vector< Solution > population;
        Solution record;
        Solution randomSolution;
        Solution tempSolution;
        std::vector< double > adjustedFitnesses;
        double sumOfFitnesses;
};

SteadyStateGA::SteadyStateGA(const TestCase& testCase
        , EvaluateFn Evaluate_
        , GenerateRandomSolutionFn GenerateRandomSolution_
        , PreprocessFn Preprocess_
        , SelectionFn Selection_
        , CrossoverFn Crossover_
        , MutationFn Mutation_
        , ReplacementFn Replacement_) : solutionLen(testCase.NumLocations),
    solutionDist(testCase.Dist),
    timeLimit(testCase.TimeLimit),
    population(PSIZE, Solution(solutionLen)),
    record(solutionLen),
    randomSolution(solutionLen),
    tempSolution(solutionLen),
    adjustedFitnesses(PSIZE),
    sumOfFitnesses(0),
    Evaluate(Evaluate_),
    GenerateRandomSolution(GenerateRandomSolution_),
    Preprocess(Preprocess_),
    Selection(Selection_),
    Crossover(Crossover_),
    Mutation(Mutation_),
    Replacement(Replacement_)
{
    for (int i = 0; i < solutionLen; ++i) {
        randomSolution.Chromosome[i] = i;
    }
}

// calculate the fitness of s and store it into s->f
void SteadyStateGA::Evaluate1(Solution& s) {
    s.Fitness = 0;
    for (int i = 0; i < solutionLen; ++i) {
        s.Fitness += solutionDist[s.Chromosome[i]][s.Chromosome[(i + 1) % solutionLen]];
    }
    if (s.Fitness < record.Fitness) {
        record = s;
    }
}

// generate a random order-based solution at s
void SteadyStateGA::GenerateRandomSolution1(Solution& s) {
    for (int i = 0; i < solutionLen; ++i) {
        int r = std::rand() % solutionLen;
        std::swap(randomSolution.Chromosome[i], randomSolution.Chromosome[r]);
    }
    std::vector< int >::iterator zeroIter = std::find(randomSolution.Chromosome.begin(), randomSolution.Chromosome.end(), 0);
    std::copy(randomSolution.Chromosome.begin(), zeroIter, std::copy(zeroIter, randomSolution.Chromosome.end(), s.Chromosome.begin()));
    // calculate the fitness
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

void SteadyStateGA::Preprocess1() {
    double maxFitness = 0;
    double minFitness = std::numeric_limits< double >::max();
    for (int i = 0; i < PSIZE; ++i) {
        maxFitness = std::max(maxFitness, population[i].Fitness);
        minFitness = std::min(minFitness, population[i].Fitness);
    }
    sumOfFitnesses = 0;
    for (int i = 0; i < PSIZE; ++i) {
        double adjustedFitness = (maxFitness - population[i].Fitness) + (maxFitness - minFitness) / (ROULETTE_SELECTION_PRESSURE_K - 1);
        sumOfFitnesses += adjustedFitness;
        adjustedFitnesses[i] = adjustedFitness;
    }
}

// choose one solution from the population
// currently this operator randomly chooses one w/ uniform Distribution
void SteadyStateGA::Selection1(Solution& p) {
    int r = std::rand() % PSIZE;
    p = population[r];
}

// Roulette Wheel - Preprocess1
void SteadyStateGA::Selection2(Solution& p) {
    double point = static_cast< double >(std::rand()) * sumOfFitnesses / RAND_MAX;
    double sum = 0;
    for (int i = 0; i < PSIZE; ++i) {
        sum += adjustedFitnesses[i];
        if (point < sum) {
            p = population[i];
            return;
        }
    }
}

// Tournament
void SteadyStateGA::Selection3(Solution& p) {
    int r1 = std::rand() % PSIZE;
    int r2 = std::rand() % PSIZE;
    const Solution& p1 = population[r1];
    const Solution& p2 = population[r2];
    double r = static_cast< double >(std::rand()) / RAND_MAX;
    if (TOURNAMENT_SELECTION_PRESSURE_T > r) {
        p = (p1.Fitness >= p2.Fitness) ? p1 : p2;
    } else {
        p = (p1.Fitness >= p2.Fitness) ? p2 : p1;
    }
}

// General Tournament
void SteadyStateGA::Selection4(Solution& p) {
    std::vector< Solution > tournament;
    for (int i = 0; i < GENERAL_TOURNAMENT_SELECTION_PRESSURE_K; ++i) {
        int r = std::rand() % PSIZE;
        tournament.push_back(population[r]);
    }
    std::sort(tournament.begin(), tournament.end());
    double r = static_cast< double >(std::rand()) / RAND_MAX;
    for (int i = 0; i < GENERAL_TOURNAMENT_SELECTION_PRESSURE_K; ++i) {
        if (TOURNAMENT_SELECTION_PRESSURE_T * std::pow((1 - TOURNAMENT_SELECTION_PRESSURE_T), i) < r) {
            p = tournament[i];
            return;
        }
    }
    p = tournament[GENERAL_TOURNAMENT_SELECTION_PRESSURE_K - 1];
}

// combine the given parents p1 and p2
// and store the generated solution at c
// currently the child will be same as p1
void SteadyStateGA::Crossover1(const Solution& p1, const Solution& p2, Solution& c) {
    int point = std::rand() % solutionLen;
    std::vector< int >::const_iterator pointIter = p1.Chromosome.begin() + point;
    std::copy(p1.Chromosome.begin(), pointIter, c.Chromosome.begin());
    int index = point;
    for (int i = 0; i < solutionLen; ++i) {
        if (std::find(p1.Chromosome.begin(), pointIter, p2.Chromosome[(i + point) % solutionLen]) == pointIter) {
            c.Chromosome[index] = p2.Chromosome[(i + point) % solutionLen];
            index++;
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// mutate the solution s
// currently this operator does nothing
void SteadyStateGA::Mutation1(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    std::swap(s.Chromosome[p], s.Chromosome[q]); // swap
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// replace one solution from the population with the new offspring
// currently any random solution can be replaced
void SteadyStateGA::Replacement1(const Solution& offspr) {
    int p = std::rand() % PSIZE;
    population[p] = offspr;
}

// a "steady-state" GA
void SteadyStateGA::GA() {
    std::time_t begin = std::time(0);
    for (int i = 0; i < PSIZE; ++i) {
        CALL_MEMBER_FN(*this, GenerateRandomSolution)(population[i]);
    }
    while (true) {
        if (std::time(0) - begin > timeLimit - 1) {
            return; // end condition
        }
        Solution p1(solutionLen);
        Solution p2(solutionLen);
        Solution c(solutionLen);
        if (Preprocess) {
            CALL_MEMBER_FN(*this, Preprocess)();
        }
        CALL_MEMBER_FN(*this, Selection)(p1);
        CALL_MEMBER_FN(*this, Selection)(p2);
        CALL_MEMBER_FN(*this, Crossover)(p1, p2, c);
        CALL_MEMBER_FN(*this, Mutation)(c);
        CALL_MEMBER_FN(*this, Replacement)(c);
    }
}

// print the best solution found to stdout
void SteadyStateGA::Answer() {
    PrintSolution(record);
}

void SteadyStateGA::PrintAllSolutions() {
    for (int i = 0; i < PSIZE; ++i) {
        PrintSolution(population[i]);
    }
}

void SteadyStateGA::PrintSolution(const Solution& s) {
    for (int i = 0; i < solutionLen; ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << s.Chromosome[i] + 1;
    }
    std::cout << " : " << s.Fitness;
    std::cout << std::endl;
}

void SteadyStateGA::Normalize(Solution& s) {
    std::copy(s.Chromosome.begin(), s.Chromosome.end(), tempSolution.Chromosome.begin());
    std::vector< int >::iterator zeroIter = std::find(tempSolution.Chromosome.begin(), tempSolution.Chromosome.end(), 0);
    std::copy(tempSolution.Chromosome.begin(), zeroIter, std::copy(zeroIter, tempSolution.Chromosome.end(), s.Chromosome.begin()));
}

int main() {
    // http://en.cppreference.com/w/cpp/numeric/random/rand
    std::srand(std::time(0));
    TestCase testCase;
    //testCase.PrintTestCase();
    SteadyStateGA ga(testCase
            , &SteadyStateGA::Evaluate1
            , &SteadyStateGA::GenerateRandomSolution1
            , NULL
            , &SteadyStateGA::Selection3
            , &SteadyStateGA::Crossover1
            , &SteadyStateGA::Mutation1
            , &SteadyStateGA::Replacement1);
    ga.GA();
    ga.Answer();
    //ga.PrintAllSolutions();
    return 0;
}
