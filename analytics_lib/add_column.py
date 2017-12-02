#add new-repeat status and add booked column
def add_column(input_set,learning_set,add_col):
    merged = input_set.merge(learning_set[['conversation_id',add_col]],on='conversation_id')
    return merged
