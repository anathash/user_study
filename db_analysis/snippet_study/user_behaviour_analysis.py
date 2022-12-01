import csv
from collections import Counter
from statistics import mean, stdev


def get_user_stats(fname):
    time_spent = []
    queries_dist = {}
    snippet_distribution = {}
    participants = {}
    age = []
    gender = []
    confidence_levels = {}

    with open(fname, newline='', encoding='utf8') as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            age.append(int(row['age']))
            gender.append(row['gender'])
            time_spent.append(float(row['search_time_exp']))
            sequence = row['sequence']
            snippets = sequence[:-1]
            alg_gen_method = sequence[-1]
            if row['confidence']:
                confidence_level = int(row['confidence'])
            else:
                confidence_level = 0
            query = row['url'].split('/')[2][:-7]

            if alg_gen_method not in participants:
                participants[alg_gen_method] = 0
            participants[alg_gen_method] += 1

            if alg_gen_method not in confidence_levels:
                confidence_levels[alg_gen_method] = []
            confidence_levels[alg_gen_method].append(confidence_level)

            if alg_gen_method not in queries_dist:
                queries_dist[alg_gen_method] = {}

            if query not in queries_dist[alg_gen_method]:
                queries_dist[alg_gen_method][query] = 0
            queries_dist[alg_gen_method][query] += 1

            if alg_gen_method not in snippet_distribution:
                snippet_distribution[alg_gen_method] = {}

            for s in snippets:
                if s not in snippet_distribution[alg_gen_method]:
                    snippet_distribution[alg_gen_method][s] = 0
                snippet_distribution[alg_gen_method][s] += 1

    c = Counter(gender)
    print('num participants: ' + str(len(age)))
    print('num participants: ' + str(len(gender)))
    print('num participants_per_extraction_method:')
    print(participants)
    print('min age:' + str(min(age)))
    print('max age:' + str(max(age)))
    print('mean age: ' + str(mean(age)))
    print('STD age: ' + str(stdev(age)))
    print('male: ' + str(c['male']) + ':' + str(c['male']/len(gender)))
    print('female: ' + str(c['female']) + ':' + str(c['female']/len(gender)))
    print('average time: ' + str(mean(time_spent)))
    print(snippet_distribution)
    print(queries_dist)

    for a, confidence in confidence_levels.items():
        print('mean confidence level for ' + a + ':' + str(mean(confidence)))
        print('STDEV confidence level for ' + a + ':' + str(stdev(confidence)))


if __name__ == "__main__":
    get_user_stats('C:\\research\\falseMedicalClaims\\user study\\WWW\\all_user_behaviour.csv')
