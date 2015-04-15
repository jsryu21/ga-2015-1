#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <iterator>
#include <set>
#include "test_case.h"

// https://isocpp.org/wiki/faq/pointers-to-members
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))

const static int PSIZE = 100; // Size of the population
const static int ROULETTE_SELECTION_PRESSURE_K = 3; // 3 ~ 4
const static double TOURNAMENT_SELECTION_PRESSURE_T = 0.6;
// http://www.complex-systems.com/pdf/09-3-2.pdf
// http://en.wikipedia.org/wiki/Tournament_selection
const static int GENERAL_TOURNAMENT_SELECTION_PRESSURE_K = 5;
const static double RANK_SELECTION_PRESSURE_MAX = 3;
const static double RANK_SELECTION_PRESSURE_MIN = 1;
const static double HYBRID_REPLACEMENT_T = 0.8;
const static double TYPICAL_MUTATION_THRESHOLD = 0.125;

struct Solution
{
    Solution(int len);
    std::vector< int > Chromosome;
    double Fitness;
    bool operator <(const Solution& solution) const {
        return Fitness < solution.Fitness;
    }
    bool operator== (const Solution& solution) const {
        return std::equal(Chromosome.begin(), Chromosome.end(), solution.Chromosome.begin());
    }
    friend std::ostream& operator<<(std::ostream& os, const Solution& solution);
};

Solution::Solution(int len) : Chromosome(std::vector< int >(len)),
    Fitness(std::numeric_limits< double >::max())
{
}

std::ostream& operator<<(std::ostream& os, const Solution& solution) {
    for (int i = 0; i < solution.Chromosome.size(); ++i) {
        os << solution.Chromosome[i] << " ";
    }
    os << ": " << solution.Fitness;
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
        typedef void (SteadyStateGA::*SelectionFn)(Solution& s);
        typedef void (SteadyStateGA::*CrossoverFn)(const Solution& p1, const Solution& p2, Solution& c);
        typedef void (SteadyStateGA::*MutationFn)(Solution& s);
        typedef void (SteadyStateGA::*ReplacementFn)(const Solution& p1, const Solution& p2, const Solution& s);
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
        void Crossover2(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover3(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover4(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover5(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover6(const Solution& p1, const Solution& p2, Solution& c);
        void Mutation1(Solution& s);
        void Mutation2(Solution& s);
        void Mutation3(Solution& s);
        void Mutation4(Solution& s);
        void Mutation5(Solution& s);
        void Mutation6(Solution& s);
        void Replacement1(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement2(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement3(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement4(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement5(const Solution& p1, const Solution& p2, const Solution& offspr);
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
        void Normalize(Solution& s);
        void InitRecords();
        void UpdateAdditoryFitnesses(int index, const Solution& s);
        void UpdateRecords(int index, const Solution& s);

        int solutionLen;
        const std::vector< std::vector< double > >& solutionDist;
        long long timeLimit;
        // population of solutions
        std::vector< Solution > population;
        Solution record;
        double maxFitness;
        int worstIndex;
        Solution randomSolution;
        Solution tempSolution;
        std::vector< double > adjustedFitnesses;
        std::vector< double > cumulativeFitnesses;
        double sumOfFitnesses;
        std::vector< bool > geneDupChecker;
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
    maxFitness(0),
    worstIndex(0),
    randomSolution(solutionLen),
    tempSolution(solutionLen),
    adjustedFitnesses(PSIZE),
    cumulativeFitnesses(PSIZE),
    sumOfFitnesses(0),
    geneDupChecker(solutionLen),
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
    int end = solutionLen - 1;
    for (int i = 0; i < end; ++i) {
        s.Fitness += solutionDist[s.Chromosome[i]][s.Chromosome[i + 1]];
    }
    s.Fitness += solutionDist[s.Chromosome[end]][s.Chromosome[0]];
}

// generate a random order-based solution at s
void SteadyStateGA::GenerateRandomSolution1(Solution& s) {
    std::random_shuffle(randomSolution.Chromosome.begin(), randomSolution.Chromosome.end());
    std::vector< int >::iterator zeroIter = std::find(randomSolution.Chromosome.begin(), randomSolution.Chromosome.end(), 0);
    std::copy(randomSolution.Chromosome.begin(), zeroIter, std::copy(zeroIter, randomSolution.Chromosome.end(), s.Chromosome.begin()));
    // calculate the fitness
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

void SteadyStateGA::Preprocess1() {
    sumOfFitnesses = 0;
    double offset = maxFitness + (maxFitness - record.Fitness) / (ROULETTE_SELECTION_PRESSURE_K - 1);
    for (int i = 0; i < PSIZE; ++i) {
        double adjustedFitness = offset - population[i].Fitness;
        sumOfFitnesses += adjustedFitness;
        adjustedFitnesses[i] = adjustedFitness;
    }
    cumulativeFitnesses[0] = adjustedFitnesses[0];
    for (int i = 1; i < PSIZE; ++i) {
        cumulativeFitnesses[i] = cumulativeFitnesses[i - 1] + adjustedFitnesses[i];
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
    std::vector< double >::iterator it = std::lower_bound(cumulativeFitnesses.begin(), cumulativeFitnesses.end(), point);
    if (it != cumulativeFitnesses.end()) {
        p = population[std::distance(cumulativeFitnesses.begin(), it)];
    } else {
        // http://valgrind.org/gallery/linux_mag.html
        p = population[PSIZE - 1];
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
    if (std::rand() % 2 == 0) {
        std::copy(p1.Chromosome.begin(), p1.Chromosome.end(), c.Chromosome.begin());
    } else {
        std::copy(p2.Chromosome.begin(), p2.Chromosome.end(), c.Chromosome.begin());
    }
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

void SteadyStateGA::Crossover2(const Solution& p1, const Solution& p2, Solution& c) {
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
void SteadyStateGA::Crossover3(const Solution& p1, const Solution& p2, Solution& c) {
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
    for (int i = 0; i < solutionLen; ++i) {
        geneDupChecker[i] = false;
    }
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
void SteadyStateGA::Crossover4(const Solution& p1, const Solution& p2, Solution& c) {
    int turn = 0;
    std::vector< bool > flags(solutionLen, false);
    std::vector< bool >::iterator iter = std::find(flags.begin(), flags.end(), false);
    while (iter != flags.end()) {
        if (turn == 0) {
            int index = std::distance(flags.begin(), iter);
            int p1Gene = p1.Chromosome[index];
            int p2Gene = p2.Chromosome[index];
            c.Chromosome[index] = p1.Chromosome[index];
            flags[index] = true;
            while (p1Gene != p2Gene) {
                index = std::distance(p1.Chromosome.begin(), std::find(p1.Chromosome.begin(), p1.Chromosome.end(), p2Gene));
                c.Chromosome[index] = p1.Chromosome[index];
                p2Gene = p2.Chromosome[index];
                flags[index] = true;
            }
            iter = std::find(flags.begin(), flags.end(), false);
        } else {
            int index = std::distance(flags.begin(), iter);
            int p2Gene = p2.Chromosome[index];
            int p1Gene = p1.Chromosome[index];
            c.Chromosome[index] = p2.Chromosome[index];
            flags[index] = true;
            while (p2Gene != p1Gene) {
                index = std::distance(p2.Chromosome.begin(), std::find(p2.Chromosome.begin(), p2.Chromosome.end(), p1Gene));
                c.Chromosome[index] = p2.Chromosome[index];
                p1Gene = p1.Chromosome[index];
                flags[index] = true;
            }
            iter = std::find(flags.begin(), flags.end(), false);
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// PMX : partially matched crossover
void SteadyStateGA::Crossover5(const Solution& p1, const Solution& p2, Solution& c) {
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
    for (int i = 0; i < p; ++i) {
        int cGene = c.Chromosome[i];
        int p2Gene = p2.Chromosome[i];
        std::vector< int >::const_iterator p1Iter = std::find(p1PIter, p1QIter, cGene);
        while (p1Iter != p1QIter) {
            p2Gene = p2.Chromosome[std::distance(p1.Chromosome.begin(), p1Iter)];
            p1Iter = std::find(p1PIter, p1QIter, p2Gene);
        }
        c.Chromosome[i] = p2Gene;
    }
    for (int i = q; i < solutionLen; ++i) {
        int cGene = c.Chromosome[i];
        int p2Gene = p2.Chromosome[i];
        std::vector< int >::const_iterator p1Iter = std::find(p1PIter, p1QIter, cGene);
        while (p1Iter != p1QIter) {
            p2Gene = p2.Chromosome[std::distance(p1.Chromosome.begin(), p1Iter)];
            p1Iter = std::find(p1PIter, p1QIter, p2Gene);
        }
        c.Chromosome[i] = p2Gene;
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// edge recombination
// http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/EdgeRecombinationCrossoverOperator.aspx
void SteadyStateGA::Crossover6(const Solution& p1, const Solution& p2, Solution& c) {
    std::vector< std::set< int > > neighborList(solutionLen);
    for (int i = 0; i < solutionLen; ++i) {
        neighborList[i].insert(p1.Chromosome[(i + solutionLen - 1) % solutionLen]);
        neighborList[i].insert(p1.Chromosome[(i + 1) % solutionLen]);
        neighborList[i].insert(p2.Chromosome[(i + solutionLen - 1) % solutionLen]);
        neighborList[i].insert(p2.Chromosome[(i + 1) % solutionLen]);
    }
    int city = 0;
    int index = 0;
    while (true) {
        c.Chromosome[index++] = city;
        if (index >= solutionLen) {
            break;
        }
        for (int i = 0; i < solutionLen; ++i) {
            neighborList[i].erase(city);
        }
        if (neighborList[city].size() == 0) {
            // random node not already in CHILD
            city = std::rand() % solutionLen;
            while (std::find(c.Chromosome.begin(), c.Chromosome.begin() + index, city) != c.Chromosome.begin() + index) {
                city = std::rand() % solutionLen;
            }
        } else {
            int fewest = 5;
            std::set< int > neighbors = neighborList[city];
            for (std::set< int >::iterator iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                if (neighborList[*iter].size() < fewest) {
                    city = *iter;
                    fewest = neighborList[*iter].size();
                }
            }
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// mutate the solution s
// two-swap or swap-change
void SteadyStateGA::Mutation1(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    std::swap(s.Chromosome[p], s.Chromosome[q]); // swap
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// typical mutation for tsp
void SteadyStateGA::Mutation2(Solution& s) {
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
void SteadyStateGA::Mutation3(Solution& s) {
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
void SteadyStateGA::Mutation4(Solution& s) {
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
void SteadyStateGA::Mutation5(Solution& s) {
    if (solutionLen < 4) {
        return;
    }
    std::vector< int > rs;
    for (int i = 0; i < 3; ++i) {
        int r = std::rand() % (solutionLen - 1);
        while (std::find(rs.begin(), rs.end(), r) != rs.end()) {
            r = std::rand() % (solutionLen - 1);
        }
        rs.push_back(r + 1);
    }
    std::sort(rs.begin(), rs.end());
    std::vector< int > mutatedGenes(solutionLen);
    std::vector< int >::iterator iter = std::copy(s.Chromosome.begin() + rs[2], s.Chromosome.end(), mutatedGenes.begin());
    iter = std::copy(s.Chromosome.begin() + rs[1], s.Chromosome.begin() + rs[2], iter);
    iter = std::copy(s.Chromosome.begin() + rs[0], s.Chromosome.begin() + rs[1], iter);
    std::copy(s.Chromosome.begin(), s.Chromosome.begin() + rs[0], iter);
    std::copy(mutatedGenes.begin(), mutatedGenes.end(), s.Chromosome.begin());
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// or change
void SteadyStateGA::Mutation6(Solution& s) {
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

// replace one solution from the population with the new offspring
// currently any random solution can be replaced
void SteadyStateGA::Replacement1(const Solution& p1, const Solution& p2, const Solution& offspr) {
    int p = std::rand() % PSIZE;

    if (Preprocess == &SteadyStateGA::Preprocess1) {
        UpdateAdditoryFitnesses(p, offspr);
    }

    UpdateRecords(p, offspr);
    population[p] = offspr;
}

// elitism
void SteadyStateGA::Replacement2(const Solution& p1, const Solution& p2, const Solution& offspr) {
    if (Preprocess == &SteadyStateGA::Preprocess1) {
        UpdateAdditoryFitnesses(worstIndex, offspr);
    }

    UpdateRecords(worstIndex, offspr);
    population[worstIndex] = offspr;
}

// preselection
void SteadyStateGA::Replacement3(const Solution& p1, const Solution& p2, const Solution& offspr) {
    std::vector< Solution >::iterator iter = std::find(population.begin(), population.end(), (p1.Fitness > p2.Fitness) ? p1 : p2);
    if (iter != population.end()) {
        int index = std::distance(population.begin(), iter);

        if (Preprocess == &SteadyStateGA::Preprocess1) {
            UpdateAdditoryFitnesses(index, offspr);
        }

        UpdateRecords(index, offspr);
        *iter = offspr;
    }
}

void SteadyStateGA::Replacement4(const Solution& p1, const Solution& p2, const Solution& offspr) {
    if (offspr.Fitness < p1.Fitness || offspr.Fitness < p2.Fitness) {
        Replacement3(p1, p2, offspr);
    } else {
        Replacement2(p1, p2, offspr);
    }
}

void SteadyStateGA::Replacement5(const Solution& p1, const Solution& p2, const Solution& offspr) {
    if (offspr.Fitness < p1.Fitness || offspr.Fitness < p2.Fitness) {
        Replacement3(p1, p2, offspr);
    }
}

// a "steady-state" GA
void SteadyStateGA::GA() {
    std::time_t begin = std::time(0);
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
        CALL_MEMBER_FN(*this, Selection)(p1);
        CALL_MEMBER_FN(*this, Selection)(p2);
        CALL_MEMBER_FN(*this, Crossover)(p1, p2, c);
        CALL_MEMBER_FN(*this, Mutation)(c);
        CALL_MEMBER_FN(*this, Replacement)(p1, p2, c);
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
    }
}

void SteadyStateGA::UpdateAdditoryFitnesses(int index, const Solution& s) {
    sumOfFitnesses -= adjustedFitnesses[index];
    double offset = maxFitness + (maxFitness - record.Fitness) / (ROULETTE_SELECTION_PRESSURE_K - 1);
    double adjustedFitness = offset - s.Fitness;
    sumOfFitnesses += adjustedFitness;
    adjustedFitnesses[index] = adjustedFitness;
    if (index == 0) {
        cumulativeFitnesses[0] = adjustedFitnesses[0];
    } else {
        cumulativeFitnesses[index] = cumulativeFitnesses[index - 1] + adjustedFitnesses[index];
    }
}

void SteadyStateGA::UpdateRecords(int index, const Solution& s) {
    if (s.Fitness < record.Fitness) {
        record = s;
    } else if (s.Fitness > maxFitness) {
        maxFitness = s.Fitness;
        worstIndex = index;
    }
}

int main(int argc, char* argv[]) {
    // http://en.cppreference.com/w/cpp/numeric/random/rand
    std::srand(std::time(0));
    TestCase testCase;
    //testCase.PrintTestCase();
    SteadyStateGA::EvaluateFn Evaluate = &SteadyStateGA::Evaluate1;
    SteadyStateGA::GenerateRandomSolutionFn GenerateRandomSolution = &SteadyStateGA::GenerateRandomSolution1;
    SteadyStateGA::PreprocessFn Preprocess = NULL;
    SteadyStateGA::SelectionFn Selection = &SteadyStateGA::Selection1;
    SteadyStateGA::CrossoverFn Crossover = &SteadyStateGA::Crossover1;
    SteadyStateGA::MutationFn Mutation = &SteadyStateGA::Mutation1;
    SteadyStateGA::ReplacementFn Replacement = &SteadyStateGA::Replacement1;
    if (argc == 5) {
        switch (*argv[1]) {
            case '0':
                Preprocess = NULL;
                Selection = &SteadyStateGA::Selection1;
                break;
            case '1':
                Preprocess = &SteadyStateGA::Preprocess1;
                Selection = &SteadyStateGA::Selection2;
                break;
            case '2':
                Preprocess = NULL;
                Selection = &SteadyStateGA::Selection3;
                break;
            case '3':
                Preprocess = NULL;
                Selection = &SteadyStateGA::Selection4;
                break;
        }
        switch (*argv[2]) {
            case '0':
                Crossover = &SteadyStateGA::Crossover1;
                break;
            case '1':
                Crossover = &SteadyStateGA::Crossover2;
                break;
            case '2':
                Crossover = &SteadyStateGA::Crossover3;
                break;
            case '3':
                Crossover = &SteadyStateGA::Crossover4;
                break;
            case '4':
                Crossover = &SteadyStateGA::Crossover5;
                break;
            case '5':
                Crossover = &SteadyStateGA::Crossover6;
                break;
        }
        switch (*argv[3]) {
            case '0':
                Mutation = &SteadyStateGA::Mutation1;
                break;
            case '1':
                Mutation = &SteadyStateGA::Mutation2;
                break;
            case '2':
                Mutation = &SteadyStateGA::Mutation3;
                break;
            case '3':
                Mutation = &SteadyStateGA::Mutation4;
                break;
            case '4':
                Mutation = &SteadyStateGA::Mutation5;
                break;
            case '5':
                Mutation = &SteadyStateGA::Mutation6;
                break;
        }
        switch (*argv[4]) {
            case '0':
                Replacement = &SteadyStateGA::Replacement1;
                break;
            case '1':
                Replacement = &SteadyStateGA::Replacement2;
                break;
            case '2':
                Replacement = &SteadyStateGA::Replacement3;
                break;
            case '3':
                Replacement = &SteadyStateGA::Replacement4;
                break;
            case '4':
                Replacement = &SteadyStateGA::Replacement5;
                break;
        }
    }
    SteadyStateGA ga(testCase
            , Evaluate
            , GenerateRandomSolution
            , Preprocess
            , Selection
            , Crossover
            , Mutation
            , Replacement);
    ga.GA();
    ga.Answer();
    //ga.PrintAllSolutions();
    return 0;
}
