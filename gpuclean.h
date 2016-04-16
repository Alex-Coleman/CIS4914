#define DEVIATIONS 3

struct line_break
{
 __host__ __device__
 bool operator()(char x)
 {
   return x == 10;
 }
};

struct white_space
{
 __host__ __device__
 bool operator()(char x)
 {
   return x != ' ';
 }
};

struct column_split
{
    char *text;         //Source csv file
    unsigned int *linebreaks;    //Location of linebreaks
    char **columns;     //Destination vectors
    int *column_width;  //Size of each entry for each vector
    int *num_columns;   //Number of columns to split
    
    column_split(char *_text, unsigned int *_linebreaks, char **_columns, int *_column_width, int *_num_columns):
    text(_text), linebreaks(_linebreaks), columns(_columns), column_width(_column_width), num_columns(_num_columns) {}
    
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        int column = 0;
        unsigned int pos = linebreaks[i];
        int j = 0;
        int t;
        
        while(column < *num_columns)
        {
            t = 0;
            j++;
            while(text[pos+j] != ',' && t < column_width[column]) {
                columns[column][column_width[column] * i + t] = text[pos+j];
                j++;
                t++;
            }
            column++;
        }
    }
};

//Search for match in keys for each value in source
//Assign value to destination
//Use binary search 
struct column_search
{
    int *keys;         //Keys being searched through, assumed to be sorted
    int *values;    //Values for each key
    int *source;     //Source keys we will be searching
    int *dest;  //Destination for values after search
    int *size;
    
    column_search(int *_keys, int *_values, int *_source, int *_dest, int *_size):
    keys(_keys), values(_values), source(_source), dest(_dest), size(_size) {}
    
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        int min = 0;
        int max = *size - 1;
        int n;
        int key = source[i];
        int value = 0;
        
        while(min <= max)
        {
            n = min + (max - min) / 2;
            if (key == keys[n]) {
                value = values[n];
                break;
            } else if (key > keys[n])
                min = n + 1;
            else 
                max = n - 1;
        }
        dest[i] = value;
    }
};

struct gpu_atoi
{
    char *source;
    int *dest;
    int *length;
    
    gpu_atoi(char *_source, int *_dest, int *_length):
        source(_source), dest(_dest), length(_length) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        int value = 0;  //integer value being determined
        int j = *length * i;    //index of character we are evaluating
        int max = *length * (i + 1);
        bool neg;             
        
        if (source[j] == '-') {
            neg = true;
            j++;
        } else {
            neg = false;
            if (source[j] == '+')
                j++;
        }        
        
        while(j < max && source[j] >= '0' && source[j] <= '9') {
            value *= 10;
            value += (source[j++] - '0'); 
        }
        if (neg){
            value = -value;
        }
        dest[i] = value;
    }
};

struct gpu_itoa
{
    int *source;
    char *dest;
    int *length;
    
    gpu_itoa(int *_source, char *_dest, int *_length):
        source(_source), dest(_dest), length(_length) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        int value = source[i];
        char text[33];
        bool isNegative = false;
        if (value < 0) {
            isNegative = true;
            value = -value;
        }

        int j = 0;
        do {
            text[j++] = value % 10 + '0';
        } while ((value /= 10) > 0);
            
        if (isNegative)
            text[j++] = '-';
        
        int k = i**length;
        int bound = (i+1)**length;
        while(j>0 && k <= bound)
            dest[k++] = text[--j];
    }
};

struct gpu_output
{
    char **source;
    char *dest;
    int *column_width;
    int *length;
    int *num_columns;
    
    gpu_output(char **_source, char *_dest, int *_column_width, int *_length, int *_num_columns):
        source(_source), dest(_dest), column_width(_column_width), length(_length), num_columns(_num_columns) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        int ind = *length * i;
        int j, k, max_k;
        for(j = 0; j < *num_columns; j++) {
            k = column_width[j] * i;
            max_k = column_width[j] * (i + 1);
            while(k < max_k && source[j][k] != NULL)
                dest[ind++] = source[j][k++];
            if ((j + 1) < *num_columns)
                dest[ind++] = ',';
        }
        dest[*length * (i + 1) - 1] = '\n';
    }
};

struct check_bounds
{
    int *source;
    char *valid;
    int *bounds;
    
    check_bounds(int *_source, char *_valid, int *_bounds):
        source(_source), valid(_valid), bounds(_bounds) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        if(source[i] < bounds[0] || source[i] > bounds[1]) {
            valid[i] = '1';
        }
    }
};

struct std_clean
{
    int *id;
    int *index;
    int *old_flow;
    int *new_flow;
    char *valid;
    int *count;
    float *global_std;
    
    std_clean(int *_id, int *_index, int *_old_flow, int *_new_flow, char *_valid, int *_count, float *_global_std):
        id(_id), index(_index), old_flow(_old_flow), new_flow(_new_flow), valid(_valid), count(_count), global_std(_global_std) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        int entry = id[i];   //lane we are testing
        int flow = 0; //flow we are testing
        int neighbors[9];       //array to hold neighbors, max 11 values
        int j = i - 1, n = 0;   //j iterates over entries, n is for neighbors
        if (valid[index[i]] == '0')
            neighbors[n++] = old_flow[index[i]];
        int n_max = n + 4;
        while (j >= 0 && n < n_max && id[j] == entry) {
            if (valid[index[j]] == '0') {
                neighbors[n] = old_flow[index[j]];
                n++;
            }
            j--;
        }
        n_max = n + 4;
        j = i + 1;
        while (j <= *count && n < n_max && id[j] == entry) {
            if (valid[index[j]] == '0') {
                neighbors[n] = old_flow[index[j]];
                n++;
            }
            j++;
        }
        //Gonna get the median, gotta uh sort it
        int temp;
        int c;
        for(int a = 0; a < n; a++) {
            c = a;
            for (int b = a + 1; b < n; b++) {
                if (neighbors[b] < neighbors[c]) {
                    c = b;
                }
            }
            temp = neighbors[a];
            neighbors[a] = neighbors[c];
            neighbors[c] = temp;
        }
        //median = n/2
        int median = neighbors[n/2];
        
        if (n != 0) {
            float mean = 0;
            for (j = 0; j < n; j++)
                mean += neighbors[j];
            mean /= n;
            if (valid[index[i]] != '0') {
                flow = (int) mean;
            } else {
                //float std = 0;
                //float temp = 0;
                /*for (j = 0; j < n; j++) {
                    temp = mean - neighbors[j];
                    std += temp * temp;
                }
                std /= n;
                std = sqrt(std);*/
                float deviation = abs(old_flow[index[i]] - median);
                if (deviation > DEVIATIONS * *global_std) {
                    flow = (int) mean;
                    valid[index[i]] = '2';
                }
                else flow = old_flow[index[i]];
            }
        }
        new_flow[index[i]] = flow;
    }
};

struct index_filler
{
    int *source;
    index_filler(int *_source):
        source(_source) {}
    
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        source[i] = i;
    }
};

struct summary_stats_data
{
    float n;
    float mean;
    float M2;

    // initialize to the identity element
    void initialize()
    {
      n = 0;
      mean = 0;
      M2 = 0;
    }

    float variance()   { return M2 / (n - 1); }
    float variance_n() { return M2 / n; }
};


struct summary_stats_unary_op
{
    int *flow;
    char *valid;
    
    summary_stats_unary_op(int *_flow, char *_valid):
    flow(_flow), valid(_valid) {}
    
    __host__ __device__
    summary_stats_data operator()(const int& x) const
    {
        summary_stats_data result;
        if (valid[x] == '0') {
         result.n    = 1;
         result.mean = (float) flow[x];
        }
        else {
         result.n = 0;
         result.mean = 0;
        }
        result.M2   = 0;

        return result;
    }
};


struct summary_stats_binary_op 
    : public thrust::binary_function<const summary_stats_data&, 
                                     const summary_stats_data&,
                                           summary_stats_data>
{
    __host__ __device__
    summary_stats_data operator()(const summary_stats_data & x, const summary_stats_data & y) const
    {
        summary_stats_data result;
            
        // precompute some common subexpressions
        if (x.n == 0 && y.n == 0) {
            result.n = 0;
            result.mean = 0;
            result.M2 = 0;
        } else {
            float n  = x.n + y.n;
            
            float delta  = y.mean - x.mean;
            float delta2 = delta  * delta;

            //Basic number of samples (n), min, and max
            result.n   = n;

            result.mean = x.mean + delta * y.n / n;

            result.M2  = x.M2 + y.M2;
            result.M2 += delta2 * x.n * y.n / n;
        }
        return result;
    }
};