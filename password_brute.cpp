#include <iostream>
#include <chrono>


using namespace std;
// to measure runtime
auto start = chrono::high_resolution_clock::now();

bool generateCombinations(const string&, string, int, string);
int breaker(string);
int main()
{
	breaker("1234");
}

int breaker(string password)
{
    string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    int pw_lenght;
    cout << "password length:";
    cin >> pw_lenght;
    if (!generateCombinations(characters, "", pw_lenght, password)) 
    {
        cout << "password not found" << endl;
    }

    // to measure runtime
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "execution time:" << elapsed.count();
    return 0;
}

//generates combinations recursively, brute force approach
bool generateCombinations(const string& characters, string current, int maxLength, string password) 
{

    //cout << "Trying: " << current << endl; //visualize recursion

    // Check for match if current is not empty
    if (!current.empty() && current == password)
    {
        cout << "Password found: " << current << endl;
        return true;
    }
    if (current.length() == maxLength)
    {
        return false;
    }

    for (char c : characters)
    {
        if (generateCombinations(characters, current + c, maxLength, password))
        {
            return true;
        }
    }
    return false;
}

