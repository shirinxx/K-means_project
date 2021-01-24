from k_means import K_means_plusplus

def main():
    
    k_means = K_means_plusplus()
    
    k_value = int(input("Enter the maximum k-value: "))
    
    
    while(k_value < 2):
        print("k-value must be equal to 2 or higher\n")
        k_value = int(input("Enter the maximum k-value: "))
    
    
    print("\n")
    
    k_means.run(k_value)
    

if __name__ == '__main__':
    main()