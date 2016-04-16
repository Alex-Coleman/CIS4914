#include <fstream>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <windows.h>
#include <math.h>
#include "gpupredict.h"
#include <curand_kernel.h>
#include <thrust/random.h>
#pragma warning(disable:4503)

#define NUMBER_OF_COLUMNS 4
#define TIMESTAMP_WIDTH 22
#define EVENT_WIDTH 32
#define LATITUDE_WIDTH 11
#define LONGITUDE_WIDTH 11
#define NUMBER_OF_BOUNDING_BOXES 300
#define NUMBER_OF_OUTPUT_COLUMNS 6
#define MONTH_WIDTH 2
#define EVENT_COUNT_WIDTH 6

using namespace std;

int main(int argc, char *argv[]) {
    //Set clock
    clock_t start = clock();
    
    //Check arguments
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <events_train>\n";
        return 0;
    }
    
    //Now we load the actual file to be cleaned
    //Windows memory mapping
    HANDLE file = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    assert(file != INVALID_HANDLE_VALUE);
    
    HANDLE fileMap = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
    assert(fileMap != INVALID_HANDLE_VALUE);
 
    LPVOID fileMapView = MapViewOfFile(fileMap, FILE_MAP_READ, 0, 0, 0);
    assert(fileMapView != NULL);

    //Copy file to GPU
    int fileSize = GetFileSize(file, NULL);
    char *fileMapViewChar = (char *)fileMapView;
    thrust::device_vector<char> fileCopy(fileSize);
    thrust::copy(fileMapViewChar, fileMapViewChar+fileSize, fileCopy.begin());
    
    //Measure linebreaks, store their location in device vector
    int linecnt = thrust::count(fileCopy.begin(), fileCopy.end(), '\n');
    thrust::device_vector<int> linebreaks(linecnt);
    thrust::counting_iterator<int> begin(0);
    thrust::copy_if(begin, begin + fileSize, fileCopy.begin(), linebreaks.begin(), line_break());
    
    //Store column widths in device vector
    thrust::device_vector<int> num_columns(1);
    num_columns[0] = NUMBER_OF_COLUMNS;
    thrust::device_vector<int> column_width(num_columns[0]);
    column_width[0] = TIMESTAMP_WIDTH;
    column_width[1] = EVENT_WIDTH;
    column_width[2] = LATITUDE_WIDTH;
    column_width[3] = LONGITUDE_WIDTH;
    
    //Create vectors for each column
    thrust::device_vector<char> tstamp(linecnt*column_width[0]);
    thrust::fill(tstamp.begin(), tstamp.end(), 0);
    thrust::device_vector<char> event(linecnt*column_width[1]);
    thrust::fill(event.begin(), event.end(), 0);
    thrust::device_vector<char> latitude_text(linecnt*column_width[2]);
    thrust::fill(latitude_text.begin(), latitude_text.end(), 0);
    thrust::device_vector<char> longitude_text(linecnt*column_width[3]);
    thrust::fill(longitude_text.begin(), longitude_text.end(), 0);
    
    //Vector to store all of the columns
    thrust::device_vector<char *> columns(num_columns[0]);
    columns[0] = thrust::raw_pointer_cast(tstamp.data());
    columns[1] = thrust::raw_pointer_cast(event.data());
    columns[2] = thrust::raw_pointer_cast(latitude_text.data());
    columns[3] = thrust::raw_pointer_cast(longitude_text.data());
    
    thrust::device_vector<int> column_locations(num_columns[0]);
    column_locations[0] = 4;
    column_locations[1] = 6;
    column_locations[2] = 9;
    column_locations[3] = 10;
    
    //Split the text into 6 columns
    column_split splitter((char *)thrust::raw_pointer_cast(fileCopy.data()), (int *)thrust::raw_pointer_cast(linebreaks.data()), (char **)thrust::raw_pointer_cast(columns.data()), (int *)thrust::raw_pointer_cast(column_width.data()), (int *)thrust::raw_pointer_cast(num_columns.data()), (int *)thrust::raw_pointer_cast(column_locations.data()));
    thrust::for_each(begin, begin + linecnt, splitter);
    
    //Convert Latitude and Longitude to floats
    thrust::device_vector<double> latitude(linecnt);
    gpu_atof latitude_tofloat((char *)thrust::raw_pointer_cast(latitude_text.data()), (double *)thrust::raw_pointer_cast(latitude.data()), (int *)thrust::raw_pointer_cast(&column_width[2]));
    thrust::for_each(begin, begin + linecnt, latitude_tofloat);  
    thrust::device_vector<double> longitude(linecnt);
    gpu_atof longitude_tofloat((char *)thrust::raw_pointer_cast(longitude_text.data()), (double *)thrust::raw_pointer_cast(longitude.data()), (int *)thrust::raw_pointer_cast(&column_width[3]));
    thrust::for_each(begin, begin + linecnt, longitude_tofloat);
    
    //Get the month from the timestamp, since that's all we need
    thrust::device_vector<int> month(linecnt);
    get_month get_months((char *)thrust::raw_pointer_cast(tstamp.data()), (int *)thrust::raw_pointer_cast(month.data()));
    thrust::for_each(begin, begin + linecnt, get_months);
    
    thrust::device_vector<double> bb_min_latitude(NUMBER_OF_BOUNDING_BOXES);
    thrust::device_vector<double> bb_max_latitude(NUMBER_OF_BOUNDING_BOXES);
    thrust::device_vector<double> bb_min_longitude(NUMBER_OF_BOUNDING_BOXES);
    thrust::device_vector<double> bb_max_longitude(NUMBER_OF_BOUNDING_BOXES);
    thrust::device_vector<int> bb_month(NUMBER_OF_BOUNDING_BOXES);
    
    create_bb bounding_boxes((double *)thrust::raw_pointer_cast(bb_min_latitude.data()), (double *)thrust::raw_pointer_cast(bb_max_latitude.data()), (double *)thrust::raw_pointer_cast(bb_min_longitude.data()), (double *)thrust::raw_pointer_cast(bb_max_longitude.data()), (int *)thrust::raw_pointer_cast(bb_month.data()));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, bounding_boxes);
    
    get_events fill_bb((double *)thrust::raw_pointer_cast(latitude.data()), (double *)thrust::raw_pointer_cast(longitude.data()), (int *)thrust::raw_pointer_cast(month.data()), (char *)thrust::raw_pointer_cast(event.data()), (double *)thrust::raw_pointer_cast(bb_min_latitude.data()), (double *)thrust::raw_pointer_cast(bb_max_latitude.data()), (double *)thrust::raw_pointer_cast(bb_min_longitude.data()), (double *)thrust::raw_pointer_cast(bb_max_longitude.data()), (int *)thrust::raw_pointer_cast(bb_month.data()));
    bb_events init;
    init.initialize();
    combine_events add_bb;
    bb_events events = thrust::transform_reduce(begin, begin + linecnt, fill_bb, init, add_bb);
    thrust::device_vector<int> events_count(NUMBER_OF_BOUNDING_BOXES);
    for(int i = 0; i < NUMBER_OF_BOUNDING_BOXES; i++)
        events_count[i] = events.events[i];
    
    //Setup for output
    thrust::device_vector<char> min_latitude_text(NUMBER_OF_BOUNDING_BOXES * LATITUDE_WIDTH);
    thrust::fill(min_latitude_text.begin(), min_latitude_text.end(), 0);
    thrust::device_vector<char> max_latitude_text(NUMBER_OF_BOUNDING_BOXES * LATITUDE_WIDTH);
    thrust::fill(max_latitude_text.begin(), max_latitude_text.end(), 0);
    thrust::device_vector<char> min_longitude_text(NUMBER_OF_BOUNDING_BOXES * LONGITUDE_WIDTH);
    thrust::fill(min_longitude_text.begin(), min_longitude_text.end(), 0);
    thrust::device_vector<char> max_longitude_text(NUMBER_OF_BOUNDING_BOXES * LONGITUDE_WIDTH);
    thrust::fill(max_longitude_text.begin(), max_longitude_text.end(), 0);
    thrust::device_vector<char> month_text(NUMBER_OF_BOUNDING_BOXES * 2);
    thrust::fill(month_text.begin(), month_text.end(), 0);
    thrust::device_vector<char> events_text(NUMBER_OF_BOUNDING_BOXES * EVENT_COUNT_WIDTH);
    thrust::fill(events_text.begin(), events_text.end(), 0);
    thrust::device_vector<int> output_num_columns(1);
    output_num_columns[0] = NUMBER_OF_OUTPUT_COLUMNS;
    thrust::device_vector<char *> output_columns(output_num_columns[0]);
    output_columns[0] = thrust::raw_pointer_cast(min_latitude_text.data());
    output_columns[1] = thrust::raw_pointer_cast(max_latitude_text.data());
    output_columns[2] = thrust::raw_pointer_cast(min_longitude_text.data());
    output_columns[3] = thrust::raw_pointer_cast(min_longitude_text.data());
    output_columns[4] = thrust::raw_pointer_cast(month_text.data());
    output_columns[5] = thrust::raw_pointer_cast(events_text.data());
    thrust::device_vector<int> output_column_width(output_num_columns[0]);
    output_column_width[0] = LATITUDE_WIDTH;
    output_column_width[1] = LATITUDE_WIDTH;
    output_column_width[2] = LONGITUDE_WIDTH;
    output_column_width[3] = LONGITUDE_WIDTH;
    output_column_width[4] = MONTH_WIDTH;
    output_column_width[5] = EVENT_COUNT_WIDTH;
    gpu_ftoa min_lat((double *)thrust::raw_pointer_cast(bb_min_latitude.data()), (char *)thrust::raw_pointer_cast(min_latitude_text.data()), (int *)thrust::raw_pointer_cast(&output_column_width[0]));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, min_lat);
    gpu_ftoa max_lat((double *)thrust::raw_pointer_cast(bb_max_latitude.data()), (char *)thrust::raw_pointer_cast(max_latitude_text.data()), (int *)thrust::raw_pointer_cast(&output_column_width[1]));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, max_lat);
    gpu_ftoa min_long((double *)thrust::raw_pointer_cast(bb_min_longitude.data()), (char *)thrust::raw_pointer_cast(min_longitude_text.data()), (int *)thrust::raw_pointer_cast(&output_column_width[2]));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, min_long);
    gpu_ftoa max_long((double *)thrust::raw_pointer_cast(bb_max_longitude.data()), (char *)thrust::raw_pointer_cast(max_longitude_text.data()), (int *)thrust::raw_pointer_cast(&output_column_width[3]));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, max_long);
    gpu_itoa month_totext((int *)thrust::raw_pointer_cast(bb_month.data()), (char *)thrust::raw_pointer_cast(month_text.data()), (int *)thrust::raw_pointer_cast(&output_column_width[4]));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, month_totext);
    gpu_itoa events_totext((int *)thrust::raw_pointer_cast(events_count.data()), (char *)thrust::raw_pointer_cast(events_text.data()), (int *)thrust::raw_pointer_cast(&output_column_width[5]));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, events_totext);
    
    thrust::copy(events_text.begin(), events_text.end(), ostream_iterator<char>(cout));
    
    thrust::device_vector<int> output_size(1);
    output_size[0] = output_column_width[0] + output_column_width[1] + output_column_width[2] + output_column_width[3] + output_column_width[4] + output_column_width[5] + output_num_columns[0];
    thrust::device_vector<char> output(NUMBER_OF_BOUNDING_BOXES * output_size[0]);
    thrust::fill(output.begin(), output.end(), 0);
    gpu_output format_output((char **)thrust::raw_pointer_cast(output_columns.data()), (char *)thrust::raw_pointer_cast(output.data()), (int *)thrust::raw_pointer_cast(output_column_width.data()), (int *)thrust::raw_pointer_cast(output_size.data()), (int *)thrust::raw_pointer_cast(output_num_columns.data()));
    thrust::for_each(begin, begin + NUMBER_OF_BOUNDING_BOXES, format_output);
    thrust::device_vector<char> final_output(output.size());
    thrust::copy_if(output.begin(), output.end(), final_output.begin(), null_space());
    thrust::device_vector<char>::iterator output_end = thrust::find(final_output.begin(), final_output.end(), NULL);
    
    //Output cleaned data
    HANDLE outputFile = CreateFileA("training_set.csv", GENERIC_WRITE | GENERIC_READ, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    assert(outputFile != INVALID_HANDLE_VALUE);

    HANDLE outputMap = CreateFileMapping(outputFile, NULL, PAGE_READWRITE, 0, output_end - final_output.begin(), NULL);
    if(outputMap == NULL)
        cout << GetLastError();        

    LPVOID outputMapView = MapViewOfFile(outputMap, FILE_MAP_WRITE, 0, 0, 0);
    
    if (outputMapView == NULL)
        cout << GetLastError();

    char *outputMapViewChar = (char *)outputMapView;
    thrust::copy(final_output.begin(), output_end, outputMapViewChar);

    cout << "Output written: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';
    
    return 0;
}