"""
Gets the frame vocabulary to be used in FAMuS that are related to Event/State/Process frames.
"""
import sys
import os
import warnings
from argparse import ArgumentParser

from nltk.corpus import framenet as fn
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from framenet_utils import get_all_relations


def get_frame_vocab(frame: str, relations: list, verbose=False):
    """
    Gets the frame relations for a specific framenet frame and list of one or more relations in the FrameNet ontology.

    Args:
        frame (str): Frame to get the relations for, the name of the frame.
        relations (list): List of relations. Format is [(relation_type, variable), (relation_type, variable), ...]
        verbose (bool, optional): Prints out the number of relations for each type. Defaults to False.

    Returns:
        frame_vocab (list): List of frame relations for the frame.
    """
    frame_vocab = []
    for relation in relations:
        frame_vocab += list(set(get_all_relations(frame, relation_type=relation[0], variable=relation[1])))
    
    frame_vocab += [frame] # adds the top of the tree
    frame_vocab = list(set(frame_vocab))
    
    if verbose:
        print(f'Frame [{frame}] has {len(frame_vocab)} related frames')

    return frame_vocab
        

def get_frame_vocab(frame, verbose=False):
    """
    Gets the frame relations for a specific framenet frame for Inheritance-Child, Predeces-Earlier/Later, and Subframe-Component.

    Args:
        frame (str): Frame to get the relations for, the name of the frame.
        verbose (bool, optional): Prints out the number of relations for each type. Defaults to False.
    
    Returns:
        frame_vocab (list): List of frame relations for the frame.
    """
    frame_vocab = []
    inherits = list(set(get_all_relations(frame))) #frame, relation_type='Inheritance', variable = 'Child'
    precedes = list(set(get_all_relations(frame, relation_type='Precedes', variable='Later'))) \
                + list(set(get_all_relations(frame, relation_type='Precedes', variable='Earlier')))
    subframe = list(set(get_all_relations(frame, relation_type='Subframe', variable='Component')))
    frame_vocab.extend(inherits)
    frame_vocab.extend(precedes)
    frame_vocab.extend(subframe)
    frame_vocab += [frame] # adds the top of the tree
    frame_vocab = list(set(frame_vocab))
    if verbose:
        print(f'Frame [{frame}] has {len(frame_vocab)} related frames: inherits {len(inherits)}, precedes {len(precedes)}, subframe {len(subframe)}')

    return frame_vocab

def remove_non_situations(frame_vocab, verbose=False):
    """
    Removes frames that are not situations from the vocab (e.g. Locale, Relation, Entity)

    Args:
        frame_vocab (list): List of frame relations for the frame.
        verbose (bool, optional): Prints out the number of relations for each type. Defaults to False.

    Returns:
        frame_vocab (list): List of frame relations for the frame.
    """

    non_situation_frames = ['Locale', 'Relation', 'Entity']
    non_situations = []
    for frame in non_situation_frames:
        non_situations += get_frame_vocab(frame)

    if verbose: vocab_length = len(frame_vocab)
    frame_vocab = [frame for frame in frame_vocab if frame not in non_situations]
    if verbose: 
        print(f'New vocabulary length: {len(frame_vocab)}. Removed {vocab_length - len(frame_vocab)} non-situation frames from the vocabulary')

    return frame_vocab

def remove_frames(frame_vocab, frames_to_remove, verbose=False):
    """
    Remove frames and their relations from the frame_vocab

    Args:
        frame_vocab (list): List of frame relations for the frame.
        frames_to_remove (list): List of frames to remove from the frame_vocab. In the format of [(frame, relation_type, variable), (frame, relation_type, variable), ...]
        verbose (bool, optional): Prints out the number of relations for each type. Defaults to False.

    Returns:
        frame_vocab (list): List of frame relations for the frame with the frames_to_remove removed.
    """

    if verbose: vocab_length = len(frame_vocab)
    for frame, relation_type, variable in frames_to_remove:
        frame_vocab = [f for f in frame_vocab if f not in get_all_relations(frame, relation_type=relation_type, variable=variable)]
    if verbose: 
        print(f'New vocabulary length: {len(frame_vocab)}. Removed {vocab_length - len(frame_vocab)} frames from the vocabulary')

    return frame_vocab



def main():
    parser = ArgumentParser()
    parser.add_argument('--frame', type=str, default=None, help='Frame to add to the vocabulary')
    parser.add_argument('--frames', type=str, default=None, help='Frames .txt file to add to the vocabulary')
    parser.add_argument('--existing-frames', type=str, default=None, help='Existing frames .txt file consisting of the current vocabulary')
    parser.add_argument('--out-file', type=str, default=None, help='Output file for vocab')
    parser.add_argument('--verbose', action='store_true', help='Verbose run')
    args = parser.parse_args()
    assert((args.frame is not None or args.frames is not None) and (args.frame is None or args.frames is None))
    
    if args.existing_frames is not None:
        with open(args.existing_frames, 'r') as f:
            existing_frames = f.read().split('\n')
            existing_frames_length = len(existing_frames)

    if args.frame is not None: #adds single frame
        frame_relations = get_frame_vocab(args.frame, verbose=args.verbose)
        if args.existing_frames is not None:
            frame_relations = list(set(frame_relations + existing_frames))
            if args.verbose: print(f'Added {len(frame_relations) - existing_frames_length} frames to the vocabulary')
        else:
            frame_relations = list(set(frame_relations))
            if args.verbose: print(f'Added {len(frame_relations)} frames to the vocabulary')

    elif args.frames is not None: #adds multiple frames
        with open(args.frames, 'r') as f:
            frames = f.read().split('\n')
        frame_relations = []
        for frame in frames:
            frame_relations += get_frame_vocab(frame, verbose=args.verbose)
        if args.existing_frames is not None:
            frame_relations = list(set(frame_relations + existing_frames))
            if args.verbose: print(f'Added {len(frame_relations) - existing_frames_length} frames to the vocabulary')
        else:
            frame_relations = list(set(frame_relations))
            if args.verbose: print(f'Added {len(frame_relations)} frames to the vocabulary')

    frame_relations = remove_non_situations(frame_relations, verbose=args.verbose)

    if args.out_file is not None:
        frame_relations.sort()
        with open(args.out_file, 'w') as f:
            for frame in frame_relations:
                f.write(frame + '\n')

if __name__ == '__main__':
    main()