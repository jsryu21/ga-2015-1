#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include "test_case.h"

// https://isocpp.org/wiki/faq/pointers-to-members
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))

const static int PSIZE = 100; // Size of the population

struct Solution
{
    Solution(int len);
    Solution(const Solution& s);
    std::vector< int > Chromosome;
    double Fitness;
};

Solution::Solution(int len) : Chromosome(std::vector< int >(len)),
    Fitness(0)
{
}

Solution::Solution(const Solution& s)
{
    this->Chromosome = s.Chromosome;
    this->Fitness = s.Fitness;
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
        typedef void (SteadyStateGA::*SelectionFn)(Solution& s);
        typedef void (SteadyStateGA::*CrossoverFn)(const Solution& p1, const Solution& p2, Solution& c);
        typedef void (SteadyStateGA::*MutationFn)(Solution& s);
        typedef void (SteadyStateGA::*ReplacementFn)(const Solution& s);
        SteadyStateGA(const TestCase& testCase
                , EvaluateFn Evaluate_
                , GenerateRandomSolutionFn GenerateRandomSolution_
                , SelectionFn Selection_
                , CrossoverFn Crossover_
                , MutationFn Mutation_
                , ReplacementFn Replacement_);
        void GA();
        void Answer();
        void PrintAllSolutions();
        void Evaluate1(Solution& s);
        void GenerateRandomSolution1(Solution& s);
        void Selection1(Solution& s);
        void Crossover1(const Solution& p1, const Solution& p2, Solution& c);
        void Mutation1(Solution& s);
        void Replacement1(const Solution& offspr);
    private:
        EvaluateFn Evaluate;
        GenerateRandomSolutionFn GenerateRandomSolution;
        SelectionFn Selection;
        CrossoverFn Crossover;
        MutationFn Mutation;
        ReplacementFn Replacement;
        void PrintSolution(const Solution& s);

        int solutionLen;
        const std::vector< std::vector< double > >& solutionDist;
        long long timeLimit;
        // population of solutions
        std::vector< Solution > population;
        Solution record;
};

SteadyStateGA::SteadyStateGA(const TestCase& testCase
        , EvaluateFn Evaluate_
        , GenerateRandomSolutionFn GenerateRandomSolution_
        , SelectionFn Selection_
        , CrossoverFn Crossover_
        , MutationFn Mutation_
        , ReplacementFn Replacement_) : solutionLen(testCase.NumLocations),
    solutionDist(testCase.Dist),
    timeLimit(testCase.TimeLimit),
    population(PSIZE, Solution(solutionLen)),
    record(solutionLen),
    Evaluate(Evaluate_),
    GenerateRandomSolution(GenerateRandomSolution_),
    Selection(Selection_),
    Crossover(Crossover_),
    Mutation(Mutation_),
    Replacement(Replacement_)
{
}

// calculate the fitness of s and store it into s->f
void SteadyStateGA::Evaluate1(Solution& s) {
    s.Fitness = 0;
    for (int i = 0; i < solutionLen; ++i) {
        s.Fitness += solutionDist[s.Chromosome[i]][s.Chromosome[(i + 1) % solutionLen]];
    }
    if (s.Fitness > record.Fitness) {
        record = s;
    }
}

// generate a random order-based solution at s
void SteadyStateGA::GenerateRandomSolution1(Solution& s) {
    for (int i = 0; i < solutionLen; ++i) {
        s.Chromosome[i] = i;
    }
    for (int i = 0; i < solutionLen; ++i) {
        int r = i + std::rand() % (solutionLen - i);	// r is a random number in [i..N-i)
        std::swap(s.Chromosome[i], s.Chromosome[r]); // swap
    }
    // calculate the fitness
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// choose one solution from the population
// currently this operator randomly chooses one w/ uniform Distribution
void SteadyStateGA::Selection1(Solution& p) {
    int r = std::rand() % PSIZE;
    p = population[r];
}

// combine the given parents p1 and p2
// and store the generated solution at c
// currently the child will be same as p1
void SteadyStateGA::Crossover1(const Solution& p1, const Solution& p2, Solution& c) {
    /*
       for (int i = 0; i < len; ++i) {
       c.Chromosome[i] = p1.Chromosome[i];
       }
       */
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// mutate the solution s
// currently this operator does nothing
void SteadyStateGA::Mutation1(Solution& s) {
    /*
       EMPTY
       */
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

void SteadyStateGA::PrintSolution(const Solution& s) {
    for (int i = 0; i < solutionLen; ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << s.Chromosome[i] + 1;
    }
    std::cout << std::endl;
}

void SteadyStateGA::PrintAllSolutions() {
    for (int i = 0; i < PSIZE; ++i) {
        PrintSolution(population[i]);
    }
}

int main() {
    // http://en.cppreference.com/w/cpp/numeric/random/rand
    std::srand(std::time(0));
    TestCase testCase;
    testCase.PrintTestCase();
    SteadyStateGA ga(testCase
            , &SteadyStateGA::Evaluate1
            , &SteadyStateGA::GenerateRandomSolution1
            , &SteadyStateGA::Selection1
            , &SteadyStateGA::Crossover1
            , &SteadyStateGA::Mutation1
            , &SteadyStateGA::Replacement1);
    ga.GA();
    ga.Answer();
    return 0;
}
