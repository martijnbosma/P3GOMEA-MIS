#include "Individual.hpp"

ostream & operator << (ostream &out, const Individual &individual)
{
	for (size_t i = 0; i < individual.numberOfVariables; ++i)
		out << +individual.genotype[i];
	out << " | " << individual.fitness << endl;
	return out;
}

void Individual::verify(int encodingLength){
    int depthArr[(int) this->genotype.size() / encodingLength + 1] = { };
    for (int i = encodingLength; i < this->genotype.size(); i++){
        if (i % encodingLength == 0) {
            while (this->genotype[i] - this->genotype[i - encodingLength] > 1) {
                this->genotype[i] = this->genotype[i] - 1;
            } 
            while (this->genotype[i] - this->genotype[i - encodingLength] < -1) {
                this->genotype[i] = this->genotype[i] + 1;
            }
            depthArr[i / encodingLength] = this->genotype[i];
        } else if (i % encodingLength == encodingLength-1) {
            int counter = std::count(depthArr, depthArr + (i / encodingLength) - 1, depthArr[i / encodingLength]);
            if (counter < this->genotype[i]) {
                this->genotype[i] = counter;
            }
        }
    }
    for (int i = this->genotype.size() - 1; i >= 0; i--){
        if (i % encodingLength == 0) {
            while (((this->genotype.size() - (i + 1)) / encodingLength) < this->genotype[i]) {
                this->genotype[i] = this->genotype[i] - 1;
            }
        }
    }
}