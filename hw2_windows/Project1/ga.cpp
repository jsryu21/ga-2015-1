#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <iterator>
#include <numeric>
#include "test_case.h"

// https://isocpp.org/wiki/faq/pointers-to-members
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))

const static int PSIZE = 100; // Size of the population
const static int ROULETTE_SELECTION_PRESSURE_K = 3; // 3 ~ 4
const static double TOURNAMENT_SELECTION_PRESSURE_T = 0.6;
// http://www.complex-systems.com/pdf/09-3-2.pdf
// http://en.wikipedia.org/wiki/Tournament_selection
const static int GENERAL_TOURNAMENT_SELECTION_PRESSURE_K = 5;
const static int GENERAL_TOURNAMENT_SELECTION_PRESSURE_T = TOURNAMENT_SELECTION_PRESSURE_T + TOURNAMENT_SELECTION_PRESSURE_T * (1 - TOURNAMENT_SELECTION_PRESSURE_T);
const static double RANK_SELECTION_PRESSURE_MAX = 3;
const static double RANK_SELECTION_PRESSURE_MIN = 1;
const static double HYBRID_REPLACEMENT_T = 0.8;
const static double TYPICAL_MUTATION_THRESHOLD = 0.125;

struct Solution
{
    Solution(int len);
    std::vector< int > Chromosome;
    double Fitness;
    friend std::ostream& operator<<(std::ostream& os, const Solution& solution);
};

Solution::Solution(int len) : Chromosome(std::vector< int >(len)),
    Fitness(std::numeric_limits< double >::max())
{
}

std::ostream& operator<<(std::ostream& os, const Solution& solution) {
    os << solution.Fitness;
    return os;
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
        typedef void (SteadyStateGA::*SelectionFn)(Solution& s, int& index);
        typedef void (SteadyStateGA::*CrossoverFn)(const Solution& p1, const Solution& p2, Solution& c);
        typedef void (SteadyStateGA::*MutationFn)(Solution& s);
        typedef void (SteadyStateGA::*LocalOptFn)(Solution& offspring, Solution& optOffSpring);
        typedef void (SteadyStateGA::*ReplacementFn)(const Solution& p1, const Solution& p2, const Solution& s, const Solution& optS, int p1Index, int p2Index);
        typedef void (SteadyStateGA::*NeedPerturbationFn)(bool& need, const Solution& s);
        typedef void (SteadyStateGA::*PerturbationFn)();
        SteadyStateGA(const std::time_t& begin_
                , const TestCase& testCase
                , EvaluateFn Evaluate_
                , GenerateRandomSolutionFn GenerateRandomSolution_
                , PreprocessFn Preprocess_
                , SelectionFn Selection_
                , CrossoverFn Crossover_
                , MutationFn Mutation_
                , LocalOptFn LocalOpt_
                , ReplacementFn Replacement_
                , NeedPerturbationFn NeedPerturbation
                , PerturbationFn Perturbation_);
        void Evaluate0(Solution& s);
        void GenerateRandomSolution0(Solution& s);
        void Preprocess0();
        void Selection0(Solution& s, int& index);
        void Selection1(Solution& s, int& index);
        void Selection2(Solution& s, int& index);
        void Selection3(Solution& s, int& index);
        void Selection4(Solution& s, int& index);
        void Crossover0(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover1(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover2(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover3(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover4(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover5(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover6(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover7(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover8(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover9(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover10(const Solution& p1, const Solution& p2, Solution& c);
        void Mutation0(Solution& s);
        void Mutation1(Solution& s);
        void Mutation2(Solution& s);
        void Mutation3(Solution& s);
        void Mutation4(Solution& s);
        void Mutation5(Solution& s);
        void Mutation6(Solution& s);
        void Mutation7(Solution& s);
        void Mutation8(Solution& s);
        void Mutation9(Solution& s);
        void Mutation10(Solution& s);
        void LocalOpt0(Solution& offspring, Solution& optOffSpring);
        void LocalOpt1(Solution& offspring, Solution& optOffSpring);
        void LocalOpt2(Solution& offspring, Solution& optOffSpring);
        void LocalOpt3(Solution& offspring, Solution& optOffSpring);
        void LocalOpt4(Solution& offspring, Solution& optOffSpring);
        void LocalOpt5(Solution& offspring, Solution& optOffSpring);
        void LocalOpt6(Solution& offspring, Solution& optOffSpring);
        void LocalOpt7(Solution& offspring, Solution& optOffSpring);
        void Replacement0(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement1(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement2(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement3(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement4(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement5(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement6(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement7(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement8(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void Replacement9(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffspr, int p1Index, int p2Index);
        void NeedPerturbation0(bool& need, const Solution& offspr);
        void NeedPerturbation1(bool& need, const Solution& offspr);
        void NeedPerturbation2(bool& need, const Solution& offspr);
        void NeedPerturbation3(bool& need, const Solution& offspr);
        void NeedPerturbation4(bool& need, const Solution& offspr);
        void NeedPerturbation5(bool& need, const Solution& offspr);
        void Perturbation0();
        void Perturbation1();
        void Perturbation2();
        void Perturbation3();
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
        LocalOptFn LocalOpt;
        ReplacementFn Replacement;
        NeedPerturbationFn NeedPerturbation;
        PerturbationFn Perturbation;
        void Normalize(Solution& s);
        void InitRecords();
        void UpdateStatistics(int index, const Solution& s);
        void UpdateAdditoryFitnesses(int index, const Solution& s);
        void UpdateRecords(int index, const Solution& s);

        std::time_t begin;
        int solutionLen;
        const std::vector< std::vector< double > >& solutionDist;
        long long timeLimit;
        // population of solutions
        std::vector< Solution > population;
        Solution record;
        int numRecordGeneration;
        double maxFitness;
        double totalFitness;
        double averageFitness;
        int worstIndex;
        Solution randomSolution;
        Solution tempSolution;
        std::vector< double > adjustedFitnesses;
        std::vector< double > cumulativeFitnesses;
        double sumOfAdjustedFitnesses;
        std::vector< bool > geneDupChecker;
        std::vector< int > corrGene;
        std::vector< std::vector< int > > cityNeighbors;
};

SteadyStateGA::SteadyStateGA(const std::time_t& begin_
        , const TestCase& testCase
        , EvaluateFn Evaluate_
        , GenerateRandomSolutionFn GenerateRandomSolution_
        , PreprocessFn Preprocess_
        , SelectionFn Selection_
        , CrossoverFn Crossover_
        , MutationFn Mutation_
        , LocalOptFn LocalOpt_
        , ReplacementFn Replacement_
        , NeedPerturbationFn NeedPerturbation_
        , PerturbationFn Perturbation_) : begin(begin_),
    solutionLen(testCase.NumLocations),
    solutionDist(testCase.Dist),
    timeLimit(testCase.TimeLimit),
    population(PSIZE, Solution(solutionLen)),
    record(solutionLen),
    numRecordGeneration(0),
    maxFitness(0),
    totalFitness(0),
    averageFitness(0),
    worstIndex(0),
    randomSolution(solutionLen),
    tempSolution(solutionLen),
    adjustedFitnesses(PSIZE),
    cumulativeFitnesses(PSIZE),
    sumOfAdjustedFitnesses(0),
    geneDupChecker(solutionLen),
    corrGene(solutionLen),
    cityNeighbors(solutionLen, std::vector< int >(4, -1)),
    Evaluate(Evaluate_),
    GenerateRandomSolution(GenerateRandomSolution_),
    Preprocess(Preprocess_),
    Selection(Selection_),
    Crossover(Crossover_),
    Mutation(Mutation_),
    LocalOpt(LocalOpt_),
    Replacement(Replacement_),
    NeedPerturbation(NeedPerturbation_),
    Perturbation(Perturbation_)
{
    for (int i = 0; i < solutionLen; ++i) {
        randomSolution.Chromosome[i] = i;
    }
}

// calculate the fitness of s and store it into s->f
void SteadyStateGA::Evaluate0(Solution& s) {
    s.Fitness = 0;
    int end = solutionLen - 1;
    for (int i = 0; i < end; ++i) {
        s.Fitness += solutionDist[s.Chromosome[i]][s.Chromosome[i + 1]];
    }
    s.Fitness += solutionDist[s.Chromosome[end]][s.Chromosome[0]];
}

// generate a random order-based solution at s
void SteadyStateGA::GenerateRandomSolution0(Solution& s) {
    std::random_shuffle(randomSolution.Chromosome.begin(), randomSolution.Chromosome.end());
    std::vector< int >::iterator zeroIter = std::find(randomSolution.Chromosome.begin(), randomSolution.Chromosome.end(), 0);
    std::copy(randomSolution.Chromosome.begin(), zeroIter, std::copy(zeroIter, randomSolution.Chromosome.end(), s.Chromosome.begin()));
    // calculate the fitness
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

void SteadyStateGA::Preprocess0() {
    sumOfAdjustedFitnesses = 0;
    double offset = maxFitness + (maxFitness - record.Fitness) / (ROULETTE_SELECTION_PRESSURE_K - 1);
    for (int i = 0; i < PSIZE; ++i) {
        double adjustedFitness = offset - population[i].Fitness;
        sumOfAdjustedFitnesses += adjustedFitness;
        adjustedFitnesses[i] = adjustedFitness;
    }
    cumulativeFitnesses[0] = adjustedFitnesses[0];
    for (int i = 1; i < PSIZE; ++i) {
        cumulativeFitnesses[i] = cumulativeFitnesses[i - 1] + adjustedFitnesses[i];
    }
}

// choose one solution from the population
// currently this operator randomly chooses one w/ uniform Distribution
void SteadyStateGA::Selection0(Solution& p, int& index) {
    index = std::rand() % PSIZE;
    p = population[index];
}

// Roulette Wheel - Preprocess0
void SteadyStateGA::Selection1(Solution& p, int& index) {
    double point = static_cast< double >(std::rand()) * sumOfAdjustedFitnesses / RAND_MAX;
    std::vector< double >::iterator it = std::lower_bound(cumulativeFitnesses.begin(), cumulativeFitnesses.end(), point);
    if (it != cumulativeFitnesses.end()) {
        index = std::distance(cumulativeFitnesses.begin(), it);
    } else {
        // http://valgrind.org/gallery/linux_mag.html
        index = PSIZE - 1;
    }
    p = population[index];
}

// Tournament
void SteadyStateGA::Selection2(Solution& p, int& index) {
    int r1 = std::rand() % PSIZE;
    int r2 = std::rand() % PSIZE;
    const Solution& p1 = population[r1];
    const Solution& p2 = population[r2];
    double r = static_cast< double >(std::rand()) / RAND_MAX;
    if (r < TOURNAMENT_SELECTION_PRESSURE_T) {
        if (p1.Fitness < p2.Fitness) {
            p = p1;
            index = r1;
        } else {
            p = p2;
            index = r1;
        }
    } else {
        if (p1.Fitness < p2.Fitness) {
            p = p2;
            index = r2;
        } else {
            p = p1;
            index = r1;
        }
    }
}

// General Tournament
void SteadyStateGA::Selection3(Solution& p, int& index) {
    double largestFitness = 0;
    double secLargestFitness = 0;
    int largestIndex = 0;
    int secLargestIndex = 0;
    int remainIndex = 0;
    for (int i = 0; i < GENERAL_TOURNAMENT_SELECTION_PRESSURE_K; ++i) {
        int r = std::rand() % PSIZE;
        double fitness = population[r].Fitness;
        if (fitness > largestFitness) {
            secLargestFitness = largestFitness;
            largestFitness = fitness;
            secLargestIndex = largestIndex;
            largestIndex = r;
        } else if (fitness > secLargestFitness) {
            secLargestFitness = fitness;
            secLargestIndex = r;
        } else {
            remainIndex = r;
        }
    }
    double r = static_cast< double >(std::rand()) / RAND_MAX;
    if (r < TOURNAMENT_SELECTION_PRESSURE_T) {
        p = population[largestIndex];
        index = largestIndex;
    } else if (r < GENERAL_TOURNAMENT_SELECTION_PRESSURE_T) {
        p = population[secLargestIndex];
        index = secLargestIndex;
    } else {
        p = population[remainIndex];
        index = remainIndex;
    }
}

void SteadyStateGA::Selection4(Solution& p, int& index) {
    int r = std::rand() % 100;
    if (r < 32) {
        Selection0(p, index);
    } else if (r < 55) {
        Selection1(p, index);
    } else if (r < 86) {
        Selection2(p, index);
    } else {
        Selection3(p, index);
    }
}

// combine the given parents p1 and p2
// and store the generated solution at c
// currently the child will be same as p1
void SteadyStateGA::Crossover0(const Solution& p1, const Solution& p2, Solution& c) {
    if (std::rand() % 2 == 0) {
        std::copy(p1.Chromosome.begin(), p1.Chromosome.end(), c.Chromosome.begin());
    } else {
        std::copy(p2.Chromosome.begin(), p2.Chromosome.end(), c.Chromosome.begin());
    }
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

void SteadyStateGA::Crossover1(const Solution& p1, const Solution& p2, Solution& c) {
    int point = std::rand() % solutionLen;
    std::vector< int >::const_iterator pointIter = p1.Chromosome.begin() + point;
    std::copy(p1.Chromosome.begin(), pointIter, c.Chromosome.begin());
    int index = point;
    bool checkExistence = 2 * point < solutionLen;
    for (int i = point; i < solutionLen; ++i) {
        if (checkExistence) {
            if (std::find(p1.Chromosome.begin(), pointIter, p2.Chromosome[i]) == pointIter) {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        } else {
            if (std::find(pointIter, p1.Chromosome.end(), p2.Chromosome[i]) != p1.Chromosome.end()) {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        }
    }
    for (int i = 0; i < point; ++i) {
        if (checkExistence) {
            if (std::find(p1.Chromosome.begin(), pointIter, p2.Chromosome[i]) == pointIter) {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        } else {
            if (std::find(pointIter, p1.Chromosome.end(), p2.Chromosome[i]) != p1.Chromosome.end()) {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        }

    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// order crossover
void SteadyStateGA::Crossover2(const Solution& p1, const Solution& p2, Solution& c) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    while (p == q) {
        q = std::rand() % solutionLen;
    }
    if (p > q) {
        std::swap(p, q);
    }
    std::vector< int >::const_iterator pIter = p1.Chromosome.begin() + p;
    std::vector< int >::const_iterator qIter = p1.Chromosome.begin() + q;
    std::copy(pIter, qIter, c.Chromosome.begin() + p);
    std::fill(geneDupChecker.begin(), geneDupChecker.end(), false);
    for (std::vector< int >::const_iterator it = pIter; it != qIter; ++it) {
        geneDupChecker[*it] = true;
    }
    int index = 0;
    for (int i = q; i < solutionLen; ++i) {
        if (geneDupChecker[p2.Chromosome[i]] == false) {
            if (p <= index && index < q) {
                c.Chromosome[q] = p2.Chromosome[i];
                index = q + 1;
            } else {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        }
    }
    for (int i = 0; i < q; ++i) {
        if (geneDupChecker[p2.Chromosome[i]] == false) {
            if (p <= index && index < q) {
                c.Chromosome[q] = p2.Chromosome[i];
                index = q + 1;
            } else {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// cycle crossover
void SteadyStateGA::Crossover3(const Solution& p1, const Solution& p2, Solution& c) {
    bool turn = true;
    std::fill(geneDupChecker.begin(), geneDupChecker.end(), false);
    std::vector< bool >::iterator it = geneDupChecker.begin();
    while (it != geneDupChecker.end()) {
        int index = std::distance(geneDupChecker.begin(), it);
        int p1Gene = p1.Chromosome[index];
        int p2Gene = p2.Chromosome[index];
        if (turn) {
            c.Chromosome[index] = p1Gene;
            geneDupChecker[index] = true;
            while (p1Gene != p2Gene) {
                index = std::distance(p1.Chromosome.begin(), std::find(p1.Chromosome.begin(), p1.Chromosome.end(), p2Gene));
                c.Chromosome[index] = p1.Chromosome[index];
                p2Gene = p2.Chromosome[index];
                geneDupChecker[index] = true;
            }
            turn = false;
        } else {
            c.Chromosome[index] = p2Gene;
            geneDupChecker[index] = true;
            while (p2Gene != p1Gene) {
                index = std::distance(p2.Chromosome.begin(), std::find(p2.Chromosome.begin(), p2.Chromosome.end(), p1Gene));
                c.Chromosome[index] = p2.Chromosome[index];
                p1Gene = p1.Chromosome[index];
                geneDupChecker[index] = true;
            }
            turn = true;
        }
        it = std::find(it, geneDupChecker.end(), false);
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// PMX : partially matched crossover
void SteadyStateGA::Crossover4(const Solution& p1, const Solution& p2, Solution& c) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    while (p == q) {
        q = std::rand() % solutionLen;
    }
    if (p > q) {
        std::swap(p, q);
    }
    std::vector< int >::const_iterator p1PIter = p1.Chromosome.begin() + p;
    std::vector< int >::const_iterator p1QIter = p1.Chromosome.begin() + q;
    std::vector< int >::const_iterator p2PIter = p2.Chromosome.begin() + p;
    std::vector< int >::const_iterator p2QIter = p2.Chromosome.begin() + q;
    std::vector< int >::iterator cIter = std::copy(p2.Chromosome.begin(), p2PIter, c.Chromosome.begin());
    cIter = std::copy(p1PIter, p1QIter, cIter);
    std::copy(p2QIter, p2.Chromosome.end(), cIter);
    std::fill(geneDupChecker.begin(), geneDupChecker.end(), false);
    for (int i = p; i < q; ++i) {
        int p1Gene = p1.Chromosome[i];
        int p2Gene = p2.Chromosome[i];
        geneDupChecker[p1Gene] = true;
        corrGene[p1Gene] = p2Gene;
    }
    for (int i = 0; i < p; ++i) {
        int& cGene = c.Chromosome[i];
        while (geneDupChecker[cGene]) {
            cGene = corrGene[cGene];
        }
    }
    for (int i = q; i < solutionLen; ++i) {
        int& cGene = c.Chromosome[i];
        while (geneDupChecker[cGene]) {
            cGene = corrGene[cGene];
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// edge recombination
// http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/EdgeRecombinationCrossoverOperator.aspx
void SteadyStateGA::Crossover5(const Solution& p1, const Solution& p2, Solution& c) {
    int end = solutionLen - 1;
    {
        int p1Prev = p1.Chromosome[end];
        int p1Next = p1.Chromosome[1];
        int p2Prev = p2.Chromosome[end];
        int p2Next = p2.Chromosome[1];
        std::vector< int >& neighbors = cityNeighbors[0];
        neighbors[0] = p1Prev;
        neighbors[1] = p1Next;
        if (p1Prev != p2Prev && p1Next != p2Prev) {
            neighbors[2] = p2Prev;
        } else {
            neighbors[2] = -1;
        }
        if (p1Prev != p2Next && p1Next != p2Next && p2Prev != p2Next) {
            neighbors[3] = p2Next;
        } else {
            neighbors[3] = -1;
        }
    }
    for (int i = 1; i < end; ++i) {
        int p1Prev = p1.Chromosome[i - 1];
        int p1Next = p1.Chromosome[i + 1];
        int p2Prev = p2.Chromosome[i - 1];
        int p2Next = p2.Chromosome[i + 1];
        std::vector< int >& neighbors = cityNeighbors[i];
        neighbors[0] = p1Prev;
        neighbors[1] = p1Next;
        if (p1Prev != p2Prev && p1Next != p2Prev) {
            neighbors[2] = p2Prev;
        } else {
            neighbors[2] = -1;
        }
        if (p1Prev != p2Next && p1Next != p2Next && p2Prev != p2Next) {
            neighbors[3] = p2Next;
        } else {
            neighbors[3] = -1;
        }
    }
    {
        int p1Prev = p1.Chromosome[end - 1];
        int p1Next = p1.Chromosome[0];
        int p2Prev = p2.Chromosome[end - 1];
        int p2Next = p2.Chromosome[0];
        std::vector< int >& neighbors = cityNeighbors[end];
        neighbors[0] = p1Prev;
        neighbors[1] = p1Next;
        if (p1Prev != p2Prev && p1Next != p2Prev) {
            neighbors[2] = p2Prev;
        } else {
            neighbors[2] = -1;
        }
        if (p1Prev != p2Next && p1Next != p2Next && p2Prev != p2Next) {
            neighbors[3] = p2Next;
        } else {
            neighbors[3] = -1;
        }
    }
    std::fill(geneDupChecker.begin(), geneDupChecker.end(), false);
    int nextCity = 0;
    int index = 0;
    while (index < solutionLen) {
        c.Chromosome[index++] = nextCity;
        geneDupChecker[nextCity] = true;
        for (int i = 0; i < solutionLen; ++i) {
            std::vector< int >& neighbors = cityNeighbors[i];
            for (int j = 0; j < 4; ++j) {
                int& neighbor = neighbors[j];
                if (neighbor == nextCity) {
                    neighbor = -1;
                }
            }
        }
        const std::vector< int >& neighbors = cityNeighbors[nextCity];
        if (neighbors[0] == -1 && neighbors[1] == -1 && neighbors[2] == -1 && neighbors[3] == -1) {
            // random node not already in CHILD
            nextCity = std::distance(geneDupChecker.begin(), std::find(geneDupChecker.begin(), geneDupChecker.end(), false));
        } else {
            int fewest = 5;
            for (int i = 0; i < 4; ++i) {
                int neighbor = neighbors[i];
                if (neighbor != -1) {
                    const std::vector< int >& farNeighbors = cityNeighbors[neighbor];
                    int count = 0;
                    for (int j = 0; j < 4; ++j) {
                        if (farNeighbors[j] != -1) {
                            count++;
                        }
                    }
                    if (count < fewest) {
                        nextCity = neighbors[i];
                        fewest = count;
                    }
                }
            }
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

void SteadyStateGA::Crossover6(const Solution& p1, const Solution& p2, Solution& c) {
    int r = std::rand() % 100;
    if (r < 25) {
        Crossover0(p1, p2, c);
    } else if (r < 38) {
        Crossover1(p1, p2, c);
    } else if (r < 73) {
        Crossover2(p1, p2, c);
    } else if (r < 84) {
        Crossover3(p1, p2, c);
    } else {
        Crossover4(p1, p2, c);
    }
}

void SteadyStateGA::Crossover7(const Solution& p1, const Solution& p2, Solution& c) {
    if (((p1.Fitness / record.Fitness) < 1.2) || ((p2.Fitness / record.Fitness) < 1.2)) {
        Crossover1(p1, p2, c);
    } else {
        Crossover0(p1, p2, c);
    }
}

void SteadyStateGA::Crossover8(const Solution& p1, const Solution& p2, Solution& c) {
    if (((p1.Fitness / record.Fitness) < 1.2) || ((p2.Fitness / record.Fitness) < 1.2)) {
        Crossover0(p1, p2, c);
    } else {
        Crossover1(p1, p2, c);
    }
}

void SteadyStateGA::Crossover9(const Solution& p1, const Solution& p2, Solution& c) {
    if (p1.Fitness < averageFitness || p2.Fitness < averageFitness) {
        Crossover1(p1, p2, c);
    } else {
        Crossover0(p1, p2, c);
    }
}

void SteadyStateGA::Crossover10(const Solution& p1, const Solution& p2, Solution& c) {
    if (p1.Fitness < averageFitness || p2.Fitness < averageFitness) {
        Crossover0(p1, p2, c);
    } else {
        Crossover1(p1, p2, c);
    }
}

// mutate the solution s
// two-swap or swap-change
void SteadyStateGA::Mutation0(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    std::swap(s.Chromosome[p], s.Chromosome[q]); // swap
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// typical mutation for tsp
void SteadyStateGA::Mutation1(Solution& s) {
    for (int i = 0; i < solutionLen; ++i) {
        double r = static_cast< double >(std::rand()) / RAND_MAX;
        if (r < TYPICAL_MUTATION_THRESHOLD) {
            int p = std::rand() % solutionLen;
            std::swap(s.Chromosome[i], s.Chromosome[p]);
        }
    }
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// range shuffle
void SteadyStateGA::Mutation2(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    if (p > q) {
        std::swap(p, q);
    }
    std::random_shuffle(s.Chromosome.begin() + p, s.Chromosome.begin() + q);
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// inversion == 2-change(two-change)
void SteadyStateGA::Mutation3(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    if (p > q) {
        std::swap(p, q);
    }
    std::reverse(s.Chromosome.begin() + p, s.Chromosome.begin() + q);
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// double bridge kick move
void SteadyStateGA::Mutation4(Solution& s) {
    if (solutionLen < 4) {
        return;
    }
    int deficitSolutionLen = solutionLen - 1;
    int r1 = (std::rand() % deficitSolutionLen) + 1;
    int r2 = (std::rand() % deficitSolutionLen) + 1;
    while (r1 == r2) {
        r2 = (std::rand() % deficitSolutionLen) + 1;
    }
    int r3 = (std::rand() % deficitSolutionLen) + 1;
    while (r1 == r3 || r2 == r3) {
        r3 = (std::rand() % deficitSolutionLen) + 1;
    }
    if (r1 > r2) {
        std::swap(r1, r2);
    }
    if (r2 > r3) {
        std::swap(r2, r3);
    }
    if (r1 > r2) {
        std::swap(r1, r2);
    }
    std::vector< int >::iterator r1It = s.Chromosome.begin() + r1;
    std::vector< int >::iterator r2It = s.Chromosome.begin() + r2;
    std::vector< int >::iterator r3It = s.Chromosome.begin() + r3;
    std::vector< int >::iterator it = std::copy(r3It, s.Chromosome.end(), tempSolution.Chromosome.begin());
    it = std::copy(r2It, r3It, it);
    it = std::copy(r1It, r2It, it);
    std::copy(s.Chromosome.begin(), r1It, it);
    std::copy(tempSolution.Chromosome.begin(), tempSolution.Chromosome.end(), s.Chromosome.begin());
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// or change
void SteadyStateGA::Mutation5(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    while (p == q) {
        q = std::rand() % solutionLen;
    }
    if (p > q) {
        std::swap(p, q);
    }
    int gene = s.Chromosome[p];
    std::rotate(s.Chromosome.begin() + p, s.Chromosome.begin() + p + 1, s.Chromosome.begin() + q + 1);
    s.Chromosome[q] = gene;
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

void SteadyStateGA::Mutation6(Solution& s) {
    int r = std::rand() % 99;
    if (r < 9) {
        Mutation0(s);
    } else if (r < 18) {
        Mutation2(s);
    } else if (r < 74) {
        Mutation3(s);
    } else if (r < 75) {
        Mutation4(s);
    } else {
        Mutation5(s);
    }
}

void SteadyStateGA::Mutation7(Solution& s) {
    if ((s.Fitness / record.Fitness) < 1.2) {
        Mutation0(s);
    } else {
        Mutation3(s);
    }
}

void SteadyStateGA::Mutation8(Solution& s) {
    if ((s.Fitness / record.Fitness) < 1.2) {
        Mutation3(s);
    } else {
        Mutation0(s);
    }
}

void SteadyStateGA::Mutation9(Solution& s) {
    if (s.Fitness < averageFitness) {
        Mutation0(s);
    } else {
        Mutation3(s);
    }
}

void SteadyStateGA::Mutation10(Solution& s) {
    if (s.Fitness < averageFitness) {
        Mutation3(s);
    } else {
        Mutation0(s);
    }
}

void SteadyStateGA::LocalOpt0(Solution& offspring, Solution& optOffSpring) {
    // http://on-demand.gputechconf.com/gtc/2014/presentations/S4534-high-speed-2-opt-tsp-solver.pdf
    while (true) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        double minChange = 0;
        for (int i = 0; i < solutionLen - 2; ++i) {
            for (int j = i + 2; j < solutionLen; ++j) {
                int jPlus = (j + 1) % solutionLen;
                double change = solutionDist[offspring.Chromosome[i]][offspring.Chromosome[j]];
                change += solutionDist[offspring.Chromosome[i + 1]][offspring.Chromosome[jPlus]];
                change -= solutionDist[offspring.Chromosome[i]][offspring.Chromosome[i + 1]];
                change -= solutionDist[offspring.Chromosome[j]][offspring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(offspring.Chromosome.begin() + reverseBegin, offspring.Chromosome.begin() + reverseEnd);
            Normalize(offspring);
            CALL_MEMBER_FN(*this, Evaluate)(offspring);
        } else {
            break;
        }
    }
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
}

void SteadyStateGA::LocalOpt1(Solution& offspring, Solution& optOffSpring) {
    while (true) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        double minChange = 0;
        bool found = false;
        for (int i = 0; (i < solutionLen - 2) && (found == false); ++i) {
            for (int j = i + 2; (j < solutionLen) && (found == false); ++j) {
                int jPlus = (j + 1) % solutionLen;
                double change = solutionDist[offspring.Chromosome[i]][offspring.Chromosome[j]];
                change += solutionDist[offspring.Chromosome[i + 1]][offspring.Chromosome[jPlus]];
                change -= solutionDist[offspring.Chromosome[i]][offspring.Chromosome[i + 1]];
                change -= solutionDist[offspring.Chromosome[j]][offspring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    found = true;
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(offspring.Chromosome.begin() + reverseBegin, offspring.Chromosome.begin() + reverseEnd);
            Normalize(offspring);
            CALL_MEMBER_FN(*this, Evaluate)(offspring);
        } else {
            break;
        }
    }
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
}

void SteadyStateGA::LocalOpt2(Solution& offspring, Solution& optOffSpring) {
    int improve = 0;
    while (improve < 20) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        double minChange = 0;
        bool found = false;
        for (int i = 0; (i < solutionLen - 2) && (found == false); ++i) {
            for (int j = i + 2; (j < solutionLen) && (found == false); ++j) {
                int jPlus = (j + 1) % solutionLen;
                double change = solutionDist[offspring.Chromosome[i]][offspring.Chromosome[j]];
                change += solutionDist[offspring.Chromosome[i + 1]][offspring.Chromosome[jPlus]];
                change -= solutionDist[offspring.Chromosome[i]][offspring.Chromosome[i + 1]];
                change -= solutionDist[offspring.Chromosome[j]][offspring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    found = true;
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(offspring.Chromosome.begin() + reverseBegin, offspring.Chromosome.begin() + reverseEnd);
            Normalize(offspring);
            CALL_MEMBER_FN(*this, Evaluate)(offspring);
            ++improve;
        } else {
            break;
        }
    }
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
}

void SteadyStateGA::LocalOpt3(Solution& offspring, Solution& optOffSpring) {
    int improve = 0;
    while (improve < 20) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        int minChange = 0;
        bool found = false;
        for (int i = 0; (i < solutionLen - 2) && (found == false); ++i) {
            for (int j = i + 2; (j < solutionLen) && (found == false); ++j) {
                int jPlus = (j + 1) % solutionLen;
                int change = solutionDist[offspring.Chromosome[i]][offspring.Chromosome[j]];
                change += solutionDist[offspring.Chromosome[i + 1]][offspring.Chromosome[jPlus]];
                change -= solutionDist[offspring.Chromosome[i]][offspring.Chromosome[i + 1]];
                change -= solutionDist[offspring.Chromosome[j]][offspring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    found = true;
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(offspring.Chromosome.begin() + reverseBegin, offspring.Chromosome.begin() + reverseEnd);
            Normalize(offspring);
            CALL_MEMBER_FN(*this, Evaluate)(offspring);
            ++improve;
        } else {
            break;
        }
    }
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
}

void SteadyStateGA::LocalOpt4(Solution& offspring, Solution& optOffSpring) {
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
    while (true) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        double minChange = 0;
        for (int i = 0; i < solutionLen - 2; ++i) {
            for (int j = i + 2; j < solutionLen; ++j) {
                int jPlus = (j + 1) % solutionLen;
                double change = solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[j]];
                change += solutionDist[optOffSpring.Chromosome[i + 1]][optOffSpring.Chromosome[jPlus]];
                change -= solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[i + 1]];
                change -= solutionDist[optOffSpring.Chromosome[j]][optOffSpring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(optOffSpring.Chromosome.begin() + reverseBegin, optOffSpring.Chromosome.begin() + reverseEnd);
            Normalize(optOffSpring);
            CALL_MEMBER_FN(*this, Evaluate)(optOffSpring);
        } else {
            break;
        }
    }
}

void SteadyStateGA::LocalOpt5(Solution& offspring, Solution& optOffSpring) {
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
    while (true) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        double minChange = 0;
        bool found = false;
        for (int i = 0; (i < solutionLen - 2) && (found == false); ++i) {
            for (int j = i + 2; (j < solutionLen) && (found == false); ++j) {
                int jPlus = (j + 1) % solutionLen;
                double change = solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[j]];
                change += solutionDist[optOffSpring.Chromosome[i + 1]][optOffSpring.Chromosome[jPlus]];
                change -= solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[i + 1]];
                change -= solutionDist[optOffSpring.Chromosome[j]][optOffSpring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    found = true;
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(optOffSpring.Chromosome.begin() + reverseBegin, optOffSpring.Chromosome.begin() + reverseEnd);
            Normalize(optOffSpring);
            CALL_MEMBER_FN(*this, Evaluate)(optOffSpring);
        } else {
            break;
        }
    }
}

void SteadyStateGA::LocalOpt6(Solution& offspring, Solution& optOffSpring) {
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
    int improve = 0;
    while (improve < 20) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        double minChange = 0;
        bool found = false;
        for (int i = 0; (i < solutionLen - 2) && (found == false); ++i) {
            for (int j = i + 2; (j < solutionLen) && (found == false); ++j) {
                int jPlus = (j + 1) % solutionLen;
                double change = solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[j]];
                change += solutionDist[optOffSpring.Chromosome[i + 1]][optOffSpring.Chromosome[jPlus]];
                change -= solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[i + 1]];
                change -= solutionDist[optOffSpring.Chromosome[j]][optOffSpring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    found = true;
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(optOffSpring.Chromosome.begin() + reverseBegin, optOffSpring.Chromosome.begin() + reverseEnd);
            Normalize(optOffSpring);
            CALL_MEMBER_FN(*this, Evaluate)(optOffSpring);
            ++improve;
        } else {
            break;
        }
    }
}

void SteadyStateGA::LocalOpt7(Solution& offspring, Solution& optOffSpring) {
    std::copy(offspring.Chromosome.begin(), offspring.Chromosome.end(), optOffSpring.Chromosome.begin());
    int improve = 0;
    while (improve < 20) {
        int reverseBegin = -1;
        int reverseEnd = -1;
        int minChange = 0;
        bool found = false;
        for (int i = 0; (i < solutionLen - 2) && (found == false); ++i) {
            for (int j = i + 2; (j < solutionLen) && (found == false); ++j) {
                int jPlus = (j + 1) % solutionLen;
                int change = solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[j]];
                change += solutionDist[optOffSpring.Chromosome[i + 1]][optOffSpring.Chromosome[jPlus]];
                change -= solutionDist[optOffSpring.Chromosome[i]][optOffSpring.Chromosome[i + 1]];
                change -= solutionDist[optOffSpring.Chromosome[j]][optOffSpring.Chromosome[jPlus]];
                if ((std::abs(minChange - change) > 0.000001) && (minChange > change)) {
                    found = true;
                    minChange = change;
                    reverseBegin = i + 1;
                    reverseEnd = j + 1;
                }
            }
        }
        if (reverseBegin != -1 && reverseEnd != -1) {
            if (reverseBegin > reverseEnd) {
                std::swap(reverseBegin, reverseEnd);
            }
            std::reverse(optOffSpring.Chromosome.begin() + reverseBegin, optOffSpring.Chromosome.begin() + reverseEnd);
            Normalize(optOffSpring);
            CALL_MEMBER_FN(*this, Evaluate)(optOffSpring);
            ++improve;
        } else {
            break;
        }
    }
}

// replace one solution from the population with the new offspring
// currently any random solution can be replaced
void SteadyStateGA::Replacement0(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    int p = std::rand() % PSIZE;
    UpdateStatistics(p, offspr);
    population[p] = offspr;
}

// elitism
void SteadyStateGA::Replacement1(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    UpdateStatistics(worstIndex, offspr);
    population[worstIndex] = offspr;
}

// preselection
void SteadyStateGA::Replacement2(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    if (p1.Fitness < p2.Fitness) {
        UpdateStatistics(p2Index, offspr);
        population[p2Index] = offspr;
    } else {
        UpdateStatistics(p1Index, offspr);
        population[p1Index] = offspr;
    }
}

void SteadyStateGA::Replacement3(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    if (optOffSpring.Fitness < p1.Fitness || optOffSpring.Fitness < p2.Fitness) {
        Replacement2(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    } else {
        Replacement1(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    }
}

void SteadyStateGA::Replacement4(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    if (optOffSpring.Fitness < p1.Fitness || optOffSpring.Fitness < p2.Fitness) {
        Replacement2(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    }
}

void SteadyStateGA::Replacement5(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    int r = std::rand() % 100;
    if (r < 10) {
        Replacement2(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    } else if (r < 57) {
        Replacement3(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    } else {
        Replacement4(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    }
}

void SteadyStateGA::Replacement6(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    if ((optOffSpring.Fitness / record.Fitness) < 1.2) {
        Replacement5(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    }
}

void SteadyStateGA::Replacement7(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    if ((optOffSpring.Fitness / record.Fitness) >= 1.2) {
        Replacement5(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    }
}

void SteadyStateGA::Replacement8(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    if (averageFitness < optOffSpring.Fitness) {
        Replacement5(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    }
}

void SteadyStateGA::Replacement9(const Solution& p1, const Solution& p2, const Solution& offspr, const Solution& optOffSpring, int p1Index, int p2Index) {
    if (averageFitness >= optOffSpring.Fitness) {
        Replacement5(p1, p2, offspr, optOffSpring, p1Index, p2Index);
    }
}

void SteadyStateGA::NeedPerturbation0(bool& need, const Solution& offspr) {
    need = false;
}

void SteadyStateGA::NeedPerturbation1(bool& need, const Solution& offspr) {
    need = ((averageFitness / record.Fitness) < 2);
}

void SteadyStateGA::NeedPerturbation2(bool& need, const Solution& offspr) {
    need = ((maxFitness / record.Fitness) < 4);
}

void SteadyStateGA::NeedPerturbation3(bool& need, const Solution& offspr) {
    double squareTotal = 0;
    for (int i = 0; i < PSIZE; ++i) {
        squareTotal += std::pow(population[i].Fitness, 2);
    }
    double dispersion = (squareTotal / PSIZE) - averageFitness * averageFitness;
    need = (dispersion < 1);
}

void SteadyStateGA::NeedPerturbation4(bool& need, const Solution& offspr) {
    need = (std::abs(record.Fitness - offspr.Fitness) <= std::numeric_limits< double >::epsilon());
}

void SteadyStateGA::NeedPerturbation5(bool& need, const Solution& offspr) {
    if (std::abs(record.Fitness - offspr.Fitness) <= std::numeric_limits< double >::epsilon()) {
        numRecordGeneration += 1;
    } else {
        numRecordGeneration += 2;
    }
    if (numRecordGeneration > 10) {
        need = true;
        numRecordGeneration = 0;
    } else {
        need = false;
    }
}

void SteadyStateGA::Perturbation0() {
    for (int i = 0; i < PSIZE; ++i) {
        if (population[i].Fitness > averageFitness) {
            CALL_MEMBER_FN(*this, GenerateRandomSolution)(population[i]);
            UpdateStatistics(i, population[i]);
        }
    }
}

void SteadyStateGA::Perturbation1() {
    for (int i = 0; i < PSIZE; ++i) {
        if (population[i].Fitness > record.Fitness) {
            CALL_MEMBER_FN(*this, GenerateRandomSolution)(population[i]);
            UpdateStatistics(i, population[i]);
        }
    }
}

void SteadyStateGA::Perturbation2() {
    double standard = (averageFitness + record.Fitness) / 2;
    for (int i = 0; i < PSIZE; ++i) {
        if (population[i].Fitness > standard) {
            CALL_MEMBER_FN(*this, GenerateRandomSolution)(population[i]);
            UpdateStatistics(i, population[i]);
        }
    }
}

void SteadyStateGA::Perturbation3() {
    double standard = (averageFitness + maxFitness) / 2;
    for (int i = 0; i < PSIZE; ++i) {
        if (population[i].Fitness > standard) {
            CALL_MEMBER_FN(*this, GenerateRandomSolution)(population[i]);
            UpdateStatistics(i, population[i]);
        }
    }
}

// a "steady-state" GA
void SteadyStateGA::GA() {
    for (int i = 0; i < PSIZE; ++i) {
        CALL_MEMBER_FN(*this, GenerateRandomSolution)(population[i]);
    }
    InitRecords();
    if (Preprocess) {
        CALL_MEMBER_FN(*this, Preprocess)();
    }
    while (true) {
        if (std::time(0) - begin > timeLimit - 1) {
            return; // end condition
        }
        Solution p1(solutionLen);
        Solution p2(solutionLen);
        Solution c(solutionLen);
        Solution optC(solutionLen);
        int p1Index;
        int p2Index;
        CALL_MEMBER_FN(*this, Selection)(p1, p1Index);
        CALL_MEMBER_FN(*this, Selection)(p2, p2Index);
        CALL_MEMBER_FN(*this, Crossover)(p1, p2, c);
        CALL_MEMBER_FN(*this, Mutation)(c);
        CALL_MEMBER_FN(*this, LocalOpt)(c, optC);
        CALL_MEMBER_FN(*this, Replacement)(p1, p2, c, optC, p1Index, p2Index);
        bool need = false;
        CALL_MEMBER_FN(*this, NeedPerturbation)(need, c);
        if (need) {
            CALL_MEMBER_FN(*this, Perturbation)();
        }
    }
}

// print the best solution found to stdout
void SteadyStateGA::Answer() {
    std::cout << record << std::endl;
}

void SteadyStateGA::PrintAllSolutions() {
    for (int i = 0; i < PSIZE; ++i) {
        std::cout << population[i] << std::endl;
    }
}

void SteadyStateGA::Normalize(Solution& s) {
    if (s.Chromosome[0] == 0) {
        return;
    }
    std::copy(s.Chromosome.begin(), s.Chromosome.end(), tempSolution.Chromosome.begin());
    std::vector< int >::iterator zeroIter = std::find(tempSolution.Chromosome.begin(), tempSolution.Chromosome.end(), 0);
    std::copy(tempSolution.Chromosome.begin(), zeroIter, std::copy(zeroIter, tempSolution.Chromosome.end(), s.Chromosome.begin()));
}

void SteadyStateGA::InitRecords() {
    for (int i = 0; i < PSIZE; ++i) {
        if (population[i].Fitness < record.Fitness) {
            record = population[i];
        } else if (population[i].Fitness > maxFitness) {
            maxFitness = population[i].Fitness;
            worstIndex = i;
        }
        totalFitness += population[i].Fitness;
    }
    averageFitness = totalFitness / PSIZE;
}

void SteadyStateGA::UpdateStatistics(int index, const Solution& s) {
    if (Preprocess == &SteadyStateGA::Preprocess0) {
        UpdateAdditoryFitnesses(index, s);
    }

    UpdateRecords(index, s);
}

void SteadyStateGA::UpdateAdditoryFitnesses(int index, const Solution& s) {
    sumOfAdjustedFitnesses -= adjustedFitnesses[index];
    double offset = maxFitness + (maxFitness - record.Fitness) / (ROULETTE_SELECTION_PRESSURE_K - 1);
    double adjustedFitness = offset - s.Fitness;
    sumOfAdjustedFitnesses += adjustedFitness;
    adjustedFitnesses[index] = adjustedFitness;
    if (index == 0) {
        cumulativeFitnesses[0] = adjustedFitnesses[0];
    } else {
        cumulativeFitnesses[index] = cumulativeFitnesses[index - 1] + adjustedFitnesses[index];
    }
}

void SteadyStateGA::UpdateRecords(int index, const Solution& s) {
    totalFitness -= population[index].Fitness;
    if (s.Fitness < record.Fitness) {
        record = s;
    } else if (s.Fitness > maxFitness) {
        maxFitness = s.Fitness;
        worstIndex = index;
    }
    totalFitness += s.Fitness;
    averageFitness = totalFitness / PSIZE;
}

int main(int argc, char* argv[]) {
    std::time_t begin = std::time(0);
    // http://en.cppreference.com/w/cpp/numeric/random/rand
    std::srand(std::time(0));
    TestCase testCase;
    //testCase.PrintTestCase();
    SteadyStateGA::EvaluateFn Evaluate = &SteadyStateGA::Evaluate0;
    SteadyStateGA::GenerateRandomSolutionFn GenerateRandomSolution = &SteadyStateGA::GenerateRandomSolution0;
    SteadyStateGA::PreprocessFn Preprocess = NULL;
    SteadyStateGA::SelectionFn Selection = &SteadyStateGA::Selection4;
    SteadyStateGA::CrossoverFn Crossover = &SteadyStateGA::Crossover6;
    SteadyStateGA::MutationFn Mutation = &SteadyStateGA::Mutation6;
    SteadyStateGA::LocalOptFn LocalOpt = &SteadyStateGA::LocalOpt0;
    SteadyStateGA::ReplacementFn Replacement = &SteadyStateGA::Replacement5;
    SteadyStateGA::NeedPerturbationFn NeedPerturbation = &SteadyStateGA::NeedPerturbation0;
    SteadyStateGA::PerturbationFn Perturbation = &SteadyStateGA::Perturbation0;
    if (argc == 8) {
        switch (atoi(argv[1])) {
            case 0:
                Preprocess = NULL;
                Selection = &SteadyStateGA::Selection0;
                break;
            case 1:
                Preprocess = &SteadyStateGA::Preprocess0;
                Selection = &SteadyStateGA::Selection1;
                break;
            case 2:
                Preprocess = NULL;
                Selection = &SteadyStateGA::Selection2;
                break;
            case 3:
                Preprocess = NULL;
                Selection = &SteadyStateGA::Selection3;
                break;
            case 4:
                Preprocess = NULL;
                Selection = &SteadyStateGA::Selection4;
        }
        switch (atoi(argv[2])) {
            case 0:
                Crossover = &SteadyStateGA::Crossover0;
                break;
            case 1:
                Crossover = &SteadyStateGA::Crossover1;
                break;
            case 2:
                Crossover = &SteadyStateGA::Crossover2;
                break;
            case 3:
                Crossover = &SteadyStateGA::Crossover3;
                break;
            case 4:
                Crossover = &SteadyStateGA::Crossover4;
                break;
            case 5:
                Crossover = &SteadyStateGA::Crossover5;
                break;
            case 6:
                Crossover = &SteadyStateGA::Crossover6;
                break;
            case 7:
                Crossover = &SteadyStateGA::Crossover7;
                break;
            case 8:
                Crossover = &SteadyStateGA::Crossover8;
                break;
            case 9:
                Crossover = &SteadyStateGA::Crossover9;
                break;
            case 10:
                Crossover = &SteadyStateGA::Crossover10;
                break;
        }
        switch (atoi(argv[3])) {
            case 0:
                Mutation = &SteadyStateGA::Mutation0;
                break;
            case 1:
                Mutation = &SteadyStateGA::Mutation1;
                break;
            case 2:
                Mutation = &SteadyStateGA::Mutation2;
                break;
            case 3:
                Mutation = &SteadyStateGA::Mutation3;
                break;
            case 4:
                Mutation = &SteadyStateGA::Mutation4;
                break;
            case 5:
                Mutation = &SteadyStateGA::Mutation5;
                break;
            case 6:
                Mutation = &SteadyStateGA::Mutation6;
                break;
            case 7:
                Mutation = &SteadyStateGA::Mutation7;
                break;
            case 8:
                Mutation = &SteadyStateGA::Mutation8;
                break;
            case 9:
                Mutation = &SteadyStateGA::Mutation9;
                break;
            case 10:
                Mutation = &SteadyStateGA::Mutation10;
                break;
        }
        switch (atoi(argv[4])) {
            case 0:
                LocalOpt = &SteadyStateGA::LocalOpt0;
                break;
            case 1:
                LocalOpt = &SteadyStateGA::LocalOpt1;
                break;
            case 2:
                LocalOpt = &SteadyStateGA::LocalOpt2;
                break;
            case 3:
                LocalOpt = &SteadyStateGA::LocalOpt3;
                break;
            case 4:
                LocalOpt = &SteadyStateGA::LocalOpt4;
                break;
            case 5:
                LocalOpt = &SteadyStateGA::LocalOpt5;
                break;
            case 6:
                LocalOpt = &SteadyStateGA::LocalOpt6;
                break;
            case 7:
                LocalOpt = &SteadyStateGA::LocalOpt7;
                break;
        }
        switch (atoi(argv[5])) {
            case 0:
                Replacement = &SteadyStateGA::Replacement0;
                break;
            case 1:
                Replacement = &SteadyStateGA::Replacement1;
                break;
            case 2:
                Replacement = &SteadyStateGA::Replacement2;
                break;
            case 3:
                Replacement = &SteadyStateGA::Replacement3;
                break;
            case 4:
                Replacement = &SteadyStateGA::Replacement4;
                break;
            case 5:
                Replacement = &SteadyStateGA::Replacement5;
                break;
            case 6:
                Replacement = &SteadyStateGA::Replacement6;
                break;
            case 7:
                Replacement = &SteadyStateGA::Replacement7;
                break;
            case 8:
                Replacement = &SteadyStateGA::Replacement8;
                break;
            case 9:
                Replacement = &SteadyStateGA::Replacement9;
                break;
        }
        switch (atoi(argv[6])) {
            case 0:
                NeedPerturbation = &SteadyStateGA::NeedPerturbation0;
                break;
            case 1:
                NeedPerturbation = &SteadyStateGA::NeedPerturbation0;
                break;
            case 2:
                NeedPerturbation = &SteadyStateGA::NeedPerturbation1;
                break;
            case 3:
                NeedPerturbation = &SteadyStateGA::NeedPerturbation2;
                break;
            case 4:
                NeedPerturbation = &SteadyStateGA::NeedPerturbation3;
                break;
            case 5:
                NeedPerturbation = &SteadyStateGA::NeedPerturbation5;
                break;
        }
        switch (atoi(argv[7])) {
            case 0:
                Perturbation = &SteadyStateGA::Perturbation0;
                break;
            case 1:
                Perturbation = &SteadyStateGA::Perturbation1;
                break;
            case 2:
                Perturbation = &SteadyStateGA::Perturbation2;
                break;
            case 3:
                Perturbation = &SteadyStateGA::Perturbation3;
                break;
        }

    }
    SteadyStateGA ga(begin
            , testCase
            , Evaluate
            , GenerateRandomSolution
            , Preprocess
            , Selection
            , Crossover
            , Mutation
            , LocalOpt
            , Replacement
            , NeedPerturbation
            , Perturbation);
    ga.GA();
    ga.Answer();
    //ga.PrintAllSolutions();
    return 0;
}
